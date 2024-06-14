import torch
import h5py
import fire
import os
import numpy as np
import datetime
from glob import glob
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn

import utils
import losses
import schedulers
from dataloader import \
    create_train_cls_dataloader,\
    create_val_cls_dataloader,\
    create_bilinear_vector_dataloader,\
    create_random_attr_dataloader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint,\
    EarlyStopping, global_step_from_engine
from ignite.metrics import RunningAverage, Loss, Average, Accuracy


class Runner(object):

    def __init__(self, seed=0):
        super(Runner, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        self.seed = seed


    
    @staticmethod
    def _forward(model, batch, device=0):
        audio_data, targets, input_ids, attn_masks = batch['audio_data'].cuda(device),\
                                               batch['target'].cuda(device),\
                                               batch['input_ids'],\
                                               batch['attention_mask']
        input_ids, attn_masks = input_ids.to(device), attn_masks.to(device)
        texts = {
            'input_ids': input_ids,
            'attention_mask': attn_masks
        }                              
        return model(audio_data, texts, targets)

    @staticmethod
    def _forward_audio(model, batch, device=0):
        waveforms, targets = batch['waveform'].cuda(device),\
                             batch['target'].cuda(device)
        return model(waveforms, targets)
    
    @staticmethod
    def _forward_vector(model, batch, eval=False, device=0):
        audio_embeds, targets = batch['audio_embed'].cuda(device),\
                                batch['target'].cuda(device)
        if eval:
            return model.calculate_score(audio_embeds, targets)  
        return model(audio_embeds, targets)  
    
    def pretrain_bilinear_vector(self,
                                 dataset: str = 'vggsound',
                                 audio_model_path: str = None,
                                 model_path: str = None,
                                 debug: bool = False,
                                 train_ratio: float = 0.8,
                                 config: str = 'config/vggsound/bilinear_vector.yaml',
                                 **kwargs):
        """
        Train using bilinear vector model
        ================================================
        Parameters
        dataset: dataset name, vggsound or audioset
        audio_model_path: path of pretrained audio model for extraing audio embeddings
        debug: whether use debug mode
        config: config file
        train_ratio: ratio of training data
        kwargs: keywords to modify configuration
        ================================================
        """

        config = utils.parse_config(
            config,
            seed=self.seed,
            debug=debug,
            **kwargs
        )
        
        classes = utils.get_classes(fold_file=config['fold_file'],
                                    select_leaveout=False,
                                    leaveout_fold=config['leaveout_fold'])
        # All classes except in leaveout fold are selected
        fold = config['leaveout_fold']

        config['audio_model_path'] = audio_model_path

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        outputdir = os.path.join(config['outputdir'], current_time)
        os.makedirs(outputdir, exist_ok=False)
        torch.save(config, os.path.join(outputdir, 'run_config.d'))
        logger = utils.Logger(os.path.join(outputdir, 'logging.txt'))

        logger.info('<============== Device Info ==============>')
        logger.info(f'Using device: {torch.cuda.get_device_name(0)}')
        logger.info('<============== Meta Data ==============>')
        logger.info(f'Output directory: {outputdir}')
        logger.info(f'Dataset: {dataset}')
        logger.info(f'Random seed: {self.seed}')
        logger.info(f'Debug mode: {debug}')
        logger.info(f'Training data ratio: {train_ratio}')
        logger.info(f'Leaveout fold: {fold}')
        logger.info(f'Number of classes: {len(classes)}')
        logger.info(f'Pretrained audio model: {audio_model_path}')
        logger.info('<============== Configuration ==============>')
        for k, v in config.items():
            logger.info(f'{k}: {v}')
        logger.info('<============== Training ==============>')

        transform = utils.get_transform()
        # get transform for audio, default numpy.array -> torch.tensor
        
        label2desc = utils.get_label2desc(desc_file=config['desc'],
                                          attr_list=config['attributes'])
        label2int = utils.get_label2int(classes=classes)
        # label2desc: dict, class -> description
        # label2int: dict, (sorted) class -> int

        indices, _ = utils.get_indices(h5=config['train_file'],
                                       classes=classes)
        print(f"Number of Samples: {len(indices)}")
        train, val = utils.split_dataset(h5=config['train_file'],
                                         label2int=label2int,
                                         train_ratio=train_ratio,
                                         indices=indices)

        train_embedding_file = os.path.join(
            audio_model_path,
            f'train_embedding_except_{config["leaveout_fold"]}.hdf5')
        val_embedding_file = os.path.join(
            audio_model_path,
            f'val_embedding_except_{config["leaveout_fold"]}.hdf5')
        audio_embeddings, train_targets = utils.get_audio_embedding(h5=config['train_file'],
                                                                    indices=train,
                                                                    pretrained_path=audio_model_path,
                                                                    audio_transform=transform,
                                                                    label2int=label2int,
                                                                    config=config.get('audio_model_kwargs', {}),
                                                                    embedding_file=train_embedding_file)
        val_audio_embeddings, val_targets = utils.get_audio_embedding(h5=config['train_file'],
                                                                      indices=val,
                                                                      pretrained_path=audio_model_path,
                                                                      config=config.get('audio_model_kwargs', {}),
                                                                      audio_transform=transform,
                                                                      label2int=label2int,
                                                                      embedding_file=val_embedding_file)
        desc_list = [label2desc[_class] for _class in sorted(classes)]
        # print(desc_list)

        tokenize_fn = utils.get_tokenize_fn(**config['tokenizer'])
        text_embeddings = utils.get_text_embedding(texts=desc_list,
                                                   tokenize_fn=tokenize_fn,
                                                   text_embedding_model=config['text_embedding_model'],
                                                   text_embedding_type=config['text_embedding_type'])
        # text_embeddings: (T, D), T: number of classes, D: dimension of text embeddings

        TrainDataloader = create_bilinear_vector_dataloader(audio_embeddings=audio_embeddings,
                                                            targets=train_targets,
                                                            is_train=True,
                                                            **config['dataloader_args'])
        ValDataloader = create_bilinear_vector_dataloader(audio_embeddings=val_audio_embeddings,
                                                          targets=val_targets,
                                                          is_train=False,
                                                          **config['dataloader_args'])
        model, *_ = utils.get_model_from_pretrain(model_path=model_path,
                                                  config=config,
                                                  audio_embed_dim=audio_embeddings.shape[-1],
                                                  text_embeddings=text_embeddings.cuda(0))
        model.cuda(0)
        output_fn = utils.get_output_func(**config['output_func_kwargs'])
        criterion = getattr(losses, config['criterion'])(output_fn=output_fn,
                                                         **config['criterion_args'])

        optimizer = getattr(torch.optim, config['optimizer'])(
            model.parameters(), **config['optimizer_args'])
        scheduler = getattr(schedulers, config['scheduler'])(
            optimizer, **config['scheduler_args'])

        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                model_out = Runner._forward_vector(model, batch, device=0)
                loss = criterion(model_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                model_out = Runner._forward_vector(
                    model, batch, eval=True, device=0)
                scores, targets = model_out['score'], model_out['target']
                # preds = torch.argmax(scores, dim=-1)
                # print((targets == preds).sum() / len(preds))
            return scores, targets
        
        trainer, evaluator = Engine(_train), Engine(_inference)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        Accuracy().attach(evaluator, 'Acc')
        ProgressBar(persist=False, ncols=75).attach(
            trainer, output_transform=lambda x: {'loss': x})
        ProgressBar(persist=False, ncols=75, desc='Evaluating').attach(
            evaluator, output_transform=lambda x: {'loss': x})
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate(engine):
            lr = optimizer.param_groups[0]['lr']
            lr = round(lr, 7)
            logger.info(f'<==== Epoch {trainer.state.epoch}, lr {lr} ====>')
            global_train_loss = engine.state.metrics['Loss']
            logger.info('Training Loss: {:<5.2f}'.format(global_train_loss))
            evaluator.run(ValDataloader)
            val_acc = evaluator.state.metrics['Acc']
            logger.info('Validation Acc: {:<5.2f}'.format(val_acc))
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, EarlyStoppingHandler)
        
        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_best', #'eval_best',
            score_function=lambda engine: evaluator.state.metrics['Acc'],
            score_name='Loss', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, 
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, BestModelCheckpoint, 
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
        )

        # trainer.run(TrainDataloader, max_epochs=config['n_epochs'])
        trainer.run(TrainDataloader,
            max_epochs=config['n_epochs'], epoch_length=config['iters_per_epoch'])
        return outputdir

    def pretrain_random_attr_vector(self,
                                    dataset: str = 'vggsound',
                                    audio_model_path: str = None,
                                    model_path: str = None,
                                    debug: bool = False,
                                    train_ratio: float = 0.8,
                                    config: str = 'config/vggsound/pretrain_random_attr_vector.yaml',
                                    **kwargs):
        """
        Train using supcon vector model as well as randomly sampling attributes

        ================================================
        Parameters
        dataset: dataset name, vggsound or audioset
        audio_model_path: path of pretrained audio model for extraing audio embeddings
        debug: whether use debug mode
        config: config file
        train_ratio: ratio of training data
        kwargs: keywords to modify configuration
        ================================================
        """

        config = utils.parse_config(
            config,
            seed=self.seed,
            debug=debug,
            **kwargs
        )
        classes = utils.get_classes(fold_file=config['fold_file'],
                                    select_leaveout=False,
                                    leaveout_fold=config['leaveout_fold'])
        fold = config['leaveout_fold']

        config['audio_model_path'] = audio_model_path

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        outputdir = os.path.join(config['outputdir'], current_time)
        os.makedirs(outputdir, exist_ok=False)
        torch.save(config, os.path.join(outputdir, 'run_config.d'))
        logger = utils.Logger(os.path.join(outputdir, 'logging.txt'))

        logger.info('<============== Device Info ==============>')
        logger.info(f'Using device: {torch.cuda.get_device_name(0)}')
        logger.info('<============== Meta Data ==============>')
        logger.info(f'Output directory: {outputdir}')
        logger.info(f'Dataset: {dataset}')
        logger.info(f'Random seed: {self.seed}')
        logger.info(f'Debug mode: {debug}')
        logger.info(f'Training data ratio: {train_ratio}')
        logger.info(f'Leaveout fold: {fold}')
        logger.info(f'Number of classes: {len(classes)}')
        logger.info(f'Pretrained audio model: {audio_model_path}')
        logger.info('<============== Configuration ==============>')
        for k, v in config.items():
            logger.info(f'{k}: {v}')
        logger.info('<============== Training ==============>')

        transform = utils.get_transform()
        # get transform for audio, default numpy.array -> torch.tensor
        
        label2desc = utils.get_label2desc(desc_file=config['desc'],
                                          attr_list=config['attributes'])
        desc_list = [label2desc[_class] for _class in sorted(classes)]
        attr_list = [desc.split('; ') for desc in desc_list]
        # desc_list (List[str]): description of (sorted) classes
        # attr_list (List[List[str]]): attributes of (sorted) classes

        label2int = utils.get_label2int(classes=classes)
        
        indices, _ = utils.get_indices(h5=config['train_file'],
                                       classes=classes)
        print(f"Number of Samples: {len(indices)}")
        train, val = utils.split_dataset(h5=config['train_file'],
                                         label2int=label2int,
                                         train_ratio=train_ratio,
                                         indices=indices)

        train_embedding_file = os.path.join(
            audio_model_path,
            f'train_embedding_except_{config["leaveout_fold"]}.hdf5')
        val_embedding_file = os.path.join(
            audio_model_path,
            f'val_embedding_except_{config["leaveout_fold"]}.hdf5')
        audio_embeddings, train_targets = utils.get_audio_embedding(h5=config['train_file'],
                                                                    indices=train,
                                                                    pretrained_path=audio_model_path,
                                                                    audio_transform=transform,
                                                                    label2int=label2int,
                                                                    config=config.get('audio_model_kwargs', {}),
                                                                    embedding_file=train_embedding_file)
        val_audio_embeddings, val_targets = utils.get_audio_embedding(h5=config['train_file'],
                                                                      indices=val,
                                                                      pretrained_path=audio_model_path,
                                                                      config=config.get('audio_model_kwargs', {}),
                                                                      audio_transform=transform,
                                                                      label2int=label2int,
                                                                      embedding_file=val_embedding_file)

        tokenize_fn = utils.get_tokenize_fn(**config['tokenizer'])
        TrainDataloader = create_random_attr_dataloader(audio_embeddings=audio_embeddings,
                                                        attr_list=attr_list,
                                                        targets=train_targets,
                                                        is_train=True,
                                                        tokenize_fn=tokenize_fn,
                                                        **config['dataloader_args'])
        ValDataloader = create_random_attr_dataloader(audio_embeddings=val_audio_embeddings,
                                                      attr_list=attr_list,
                                                      targets=val_targets,
                                                      is_train=False,
                                                      tokenize_fn=tokenize_fn,
                                                      **config['dataloader_args'])
        model, *_ = utils.get_model_from_pretrain(model_path=model_path,
                                                  config=config,
                                                  audio_embed_dim=audio_embeddings.shape[-1])
        model.cuda(0)
        output_fn = utils.get_output_func(**config['output_func_kwargs'])
        criterion = getattr(losses, config['criterion'])(output_fn=output_fn,
                                                         **config['criterion_args'])

        optimizer = getattr(torch.optim, config['optimizer'])(
            model.parameters(), **config['optimizer_args'])
        scheduler = getattr(schedulers, config['scheduler'])(
            optimizer, **config['scheduler_args'])

        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                model_out = Runner._forward(
                    model, batch, device=0)
                loss = criterion(model_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.item()
        
        
        trainer = Engine(_train)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        ProgressBar(persist=False, ncols=75).attach(
            trainer, output_transform=lambda x: {'loss': x})
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def check_training(engine):
            lr = optimizer.param_groups[0]['lr']
            lr = round(lr, 7)
            logger.info(f'<==== Epoch {trainer.state.epoch}, lr {lr} ====>')
            global_train_loss = engine.state.metrics['Loss']
            logger.info('Training Loss: {:<5.2f}'.format(global_train_loss))
            val_acc = evaluate()
            engine.state.metrics['Acc'] = val_acc
            logger.info('Evaluation Acc: {:<5.2f}'.format(val_acc))
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        @torch.no_grad()
        def evaluate():
            model.eval()
            text_embeddings = {}
            audio_embeddings, ground_truths = [], []
            for batch in tqdm(ValDataloader, desc='Evaluating', ncols=85):
                model_out = Runner._forward(model, batch, device=0)
                ground_truths.append(model_out['audio_target'].cpu())
                audio_embeddings.append(model_out['audio_proj'])

                # text_embeddings (dict): text_target -> text_embed
                # text_embed from all attributes
                for text_embed, text_target in zip(model_out['text_proj'], model_out['text_target']):
                    text_embeddings[text_target.cpu().item()] = text_embed
            audio_embeddings = torch.cat(audio_embeddings, dim=0).cpu()
            ground_truths = torch.cat(ground_truths, dim=0).numpy()
            text_targets = torch.tensor(list(text_embeddings.keys()))
            text_embeddings = torch.stack(list(text_embeddings.values()), dim=0).cpu()
            scores = audio_embeddings @ text_embeddings.T
            predictions = text_targets[torch.argmax(scores, dim=1)].numpy()
            acc = (predictions == ground_truths).sum() / len(predictions)
            return acc


        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, EarlyStoppingHandler)
        
        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_best', #'eval_best',
            score_function=lambda engine: engine.state.metrics['Acc'],
            score_name='Acc', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, 
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, BestModelCheckpoint, 
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
        )
        trainer.run(TrainDataloader,
            max_epochs=config['n_epochs'], epoch_length=config['iters_per_epoch'])
        # trainer.run(TrainDataloader, max_epochs=config['n_epochs'])

        return outputdir


    @torch.no_grad()
    def zero_shot_bilinear_vector(self,
                                  model_path: str,
                                  test_h5: str = 'vggsound_test_classname.hdf5'):
        """
        Zero-shot using baseline method
        Perform zero-shot on leaveout fold
        Get configuration from config file stored in model_path
        Get pretrained audio-text model from model_path

        ============================
        Parameters:
        model_path: path of the model
        dataset: dataset name
        test_h5: path of the test h5 file
        """

        path = Path(model_path)
        assert path.exists()

        ############## config ##############
        config_file = list(path.glob('run_config.d'))[0]
        config = torch.load(config_file, map_location='cpu')
        import json
        pretty_config = json.dumps(config, indent=4)
        print("############## Configuration ##############")
        print(pretty_config)
        print("############## Configuration ##############")
        
        test_h5 = test_h5 or config['train_file']
        label2desc = utils.get_label2desc(desc_file=config['desc'],
                                          attr_list=config['attributes'])
        classes = utils.get_classes(fold_file=config['fold_file'],
                                    select_leaveout=True,
                                    leaveout_fold=config['leaveout_fold'])
        classes = sorted(classes)

        label2int = utils.get_label2int(classes=classes)
        desc_list = [label2desc[_class] for _class in sorted(classes)]
        test, _ = utils.get_indices(h5=test_h5,
                                    classes=classes)
        print(f'Number of classes in fold {config["leaveout_fold"]}: {len(classes)}')
        print(f"Number of test samples: {len(test)}")
        transform = utils.get_transform()

        tokenize_fn = utils.get_tokenize_fn(**config['tokenizer'])
        text_embeddings = utils.get_text_embedding(texts=desc_list,
                                                   tokenize_fn=tokenize_fn,
                                                   text_embedding_model=config['text_embedding_model'],
                                                   text_embedding_type=config['text_embedding_type'])

        embedding_file = os.path.join(config['audio_model_path'],
                                      f'test_embedding_only_{config["leaveout_fold"]}.hdf5')
        audio_embeddings, targets = utils.get_audio_embedding(h5=test_h5,
                                                              indices=test,
                                                              pretrained_path=config['audio_model_path'],
                                                              audio_transform=transform,
                                                              label2int=label2int,
                                                              config=config.get('audio_model_kwargs', {}),
                                                              embedding_file=embedding_file)

        model, _, _ = utils.get_model_from_pretrain(model_path=model_path,
                                                    config=config,
                                                    resume=True,
                                                    audio_embed_dim=audio_embeddings.shape[-1],
                                                    text_embeddings=text_embeddings.cuda(0))
        model.eval().cuda(0)

        scores = model.calculate_score(audio_embeddings.cuda(0), targets)['score'].cpu()
        # scores: (B, T), B: batch_size, T: number of class
        predictions = torch.argmax(scores, dim=-1).tolist()
        ground_truths = targets

        from sklearn.metrics import accuracy_score, classification_report
        import pandas as pd
        acc = accuracy_score(ground_truths, predictions)
        report = classification_report(ground_truths, predictions,
                                       target_names=classes,
                                       output_dict=True)
        report_df = pd.DataFrame(report).T
        report_df.to_csv(os.path.join(model_path, 'report.csv'))
        with open(os.path.join(model_path, 'acc.txt'), 'w') as output:
            output.write(f'Accuracy is {acc}')
        # print(acc)
        return acc

    @torch.no_grad()
    def zero_shot_random_attr_vector(self,
                                     model_path: str,
                                     test_h5: str = 'vggsound_test_classname.hdf5'):
        """
        Zero-shot using Proposed method
        Perform zero-shot on leaveout fold
        Get configuration from config file stored in model_path
        Get pretrained audio-text model from model_path

        ============================
        Parameters:
        model_path: path of the model
        test_h5: path of the test h5 file
        """
        path = Path(model_path)
        assert path.exists()


        ############## config ##############
        config_file = list(path.glob('run_config.d'))[0]
        config = torch.load(config_file, map_location='cpu')


        import json
        pretty_config = json.dumps(config, indent=4)
        print("############## Configuration ##############")
        print(pretty_config)
        print("############## Configuration ##############")

        ############## test data ##############
        test_h5 = test_h5 or config['train_file']
        label2desc = utils.get_label2desc(desc_file=config['desc'],
                                          attr_list=config['attributes'])
        classes = utils.get_classes(fold_file=config['fold_file'],
                                    select_leaveout=True,
                                    leaveout_fold=config['leaveout_fold'])
        classes = sorted(classes)
        test, _ = utils.get_indices(h5=test_h5,
                                    classes=classes)
        print(f'Number of classes in fold {config["leaveout_fold"]}: {len(classes)}')
        print(f"Number of test samples: {len(test)}")

        ############## audio embedding ##############
        label2int = utils.get_label2int(classes=classes)
        desc_list = [label2desc[_class] for _class in classes]
        transform = utils.get_transform()

        embedding_file = os.path.join(
            config['audio_model_path'],
            f'test_embedding_only_{config["leaveout_fold"]}.hdf5')
        audio_embeddings, targets = utils.get_audio_embedding(h5=test_h5,
                                                              indices=test,
                                                              pretrained_path=config['audio_model_path'],
                                                              audio_transform=transform,
                                                              label2int=label2int,
                                                              config=config.get('audio_model_kwargs', {}),
                                                              embedding_file=embedding_file)
        ################ text embedding ##############
        tokenize_fn = utils.get_tokenize_fn(**config['tokenizer'])
        texts = tokenize_fn(desc_list)
        # texts are tokenized descriptions of classes
        for k, v in texts.items():
            texts[k] = v.cuda(0)

        ################ model load ##############
        model, _, _ = utils.get_model_from_pretrain(model_path=model_path,
                                                    config=config,
                                                    resume=True,
                                                    audio_embed_dim=audio_embeddings.shape[-1])
        model.eval().cuda(0)

        ################ encode ##############
        audio_embeds = model.encode_audio(audio_embeddings.cuda(0))
        text_embeds = model.encode_text(texts)
        
        ################ prediction ##############
        import torch.nn as nn
        audio_embeds = nn.functional.normalize(audio_embeds, dim=-1)
        text_embeds = nn.functional.normalize(text_embeds, dim=-1)

        scores = (audio_embeds @ text_embeds.T).cpu()
        predictions = torch.argmax(scores, dim=1).numpy()

        from sklearn.metrics import accuracy_score, classification_report, f1_score
        import pandas as pd
        acc = accuracy_score(targets, predictions)
        f1_macro = f1_score(targets, predictions, average='macro')
        f1_micro = f1_score(targets, predictions, average='micro')
        report = classification_report(targets, predictions,
                                       target_names=classes,
                                       output_dict=True)
        report_df = pd.DataFrame(report).T
        report_df.to_csv(os.path.join(model_path, 'report.csv'))
        with open(os.path.join(model_path, 'acc.txt'), 'w') as output:
            output.write(f'Accuracy is {acc}')
            output.write(f'f1_macro is {f1_macro}')
            output.write(f'f1_micro is {f1_micro}')


        return {
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }


    @torch.no_grad()
    def batch_zero_shot_bilinear_vector(self,
                                        base_dir: str,
                                        test_h5: str = 'vggsound_test_classname.hdf5'):
        """
        Evaluate Baseline zero-shot on multiple models
        models from base_dir/Seed*/*/

        ================================================
        Parameters:
        base_dir: base directory of models
        test_h5: path of the test h5 file
        """
        from glob import glob
        folders = glob(os.path.join(base_dir, 'Seed*/*/'))
        accs = []
        for folder in folders:
            print(f"<=========== {folder} starts ===========>")
            acc = self.zero_shot_bilinear_vector(model_path=folder,
                                                 test_h5=test_h5)
            accs.append(acc)
            print(f"<=========== {folder} finished ===========>")
        accs = np.array(accs)
        with open(os.path.join(base_dir, 'acc.txt'), 'w') as output:
            output.write(f'Mean accuracy is {np.mean(accs)}\n')
            output.write(f'Std accuracy is {np.std(accs)}\n')
            output.write(f'Accuracy: {accs}\n')
        print(f'Mean accuracy is {np.mean(accs)}')
        print(f'Std accuracy is {np.std(accs)}')

    @torch.no_grad()
    def batch_zero_shot_random_attr(self,
                                    base_dir: str,
                                    test_h5: str = 'vggsound_test_classname.hdf5'):
        """
        Evaluate Proposed method using zero-shot on multiple models
        models from base_dir/Seed*/*/

        ================================================
        Parameters:
        base_dir: base directory of models
        test_h5: path of the test h5 file
        """

        from glob import glob
        folders = glob(os.path.join(base_dir, 'Seed*/*/'))
        metrics = {}
        for folder in folders:
            metric = self.zero_shot_random_attr_vector(model_path=folder,
                                                       test_h5=test_h5)
            for name, value in metric.items():
                metrics.setdefault(name, []).append(value)
        with open(os.path.join(base_dir, 'acc.txt'), 'w') as output:
            for name, values in metrics.items():
                output.write(f'Mean {name} is {np.mean(values)}\n')
                output.write(f'Std {name} is {np.std(values)}\n')
                print(f'Mean {name} is {np.mean(values)}')
                print(f'Std {name} is {np.std(values)}')
                output.write(f'{name}: {values}\n')


    def train(self,
              model_path: str = None,
              debug: bool = False,
              resume: bool = False,
              config: str = 'config/vggsound/train.yaml',
              train_ratio: float = 0.8,
              **kwargs):
        """
        Audio classification with one fold leaveout

        ================================================
        Parameters
        config: config file
        debug: whether to use debug mode
        resume: whether to resume training from <model_path>
        model_path: path of the pretrained model
        train_ratio: ratio of training data
        kwargs: keywords to modify configuration
        ================================================
        """
        if resume:
            assert model_path is not None
            config_file = glob(os.path.join(model_path, 'run_config.d'))[0]
            config = torch.load(config_file, map_location='cpu')
            for k, v in kwargs.items():
                config[k] = v
            if config['scheduler'] == 'CosineAnnealingLRReduce':
                config['scheduler_args']['warmup'] = False
        else:
            config = utils.parse_config(
                config,
                seed=self.seed,
                debug=debug,
                **kwargs
            )
        classes = utils.get_classes(fold_file=config['fold_file'],
                                    select_leaveout=False,
                                    leaveout_fold=config['leaveout_fold'])
        config['model_kwargs']['n_class'] = len(classes)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fold = config['leaveout_fold']
        outputdir = os.path.join(config['outputdir'], fold, current_time)
        if resume:
            outputdir = os.path.join(model_path, 'Resume', current_time)
        os.makedirs(outputdir, exist_ok=False)
        torch.save(config, os.path.join(outputdir, 'run_config.d'))
        logger = utils.Logger(os.path.join(outputdir, 'logging.txt'))

        logger.info('<============== Device Info ==============>')
        logger.info(f'Using device: {torch.cuda.get_device_name(0)}')
        logger.info('<============== Meta Data ==============>')
        if resume:
            logger.info(f'Resume training from {model_path}')
    
        logger.info(f'Output directory is: {outputdir}')
        logger.info(f'Random seed: {self.seed}')
        logger.info(f'Debug mode: {debug}')
        logger.info(f'Training data ratio: {train_ratio}')
        logger.info(f'Leaveout fold: {fold}')
        logger.info(f'Number of classes: {len(classes)}')
        logger.info('<============== Configuration ==============>')
        for k, v in config.items():
            logger.info(f'{k}: {v}')
        logger.info('<============== Training ==============>')

        train_transform = utils.get_transform()
        val_transform = utils.get_transform()
        
        indices, _ = utils.get_indices(h5=config['train_file'],
                                       classes=classes)
        label2int = utils.get_label2int(classes=classes)
        train, val = utils.split_dataset(h5=config['train_file'],
                                         train_ratio=train_ratio,
                                         label2int=label2int,
                                         indices=indices)
        print(f"Number of training samples: {len(train)}")
        print(f"Number of validation samples: {len(val)}")
        TrainDataloader = create_train_cls_dataloader(audio_file=config['train_file'],
                                                      indices=train,
                                                      label2int=label2int,
                                                      audio_transform=train_transform,
                                                      **config['train_dataloader_args'])
        ValDataloader = create_val_cls_dataloader(audio_file=config['train_file'],
                                                  indices=val,
                                                  label2int=label2int,
                                                  audio_transform=val_transform,
                                                  **config['val_dataloader_args'])
        
        model, optimizer_params, scheduler_params =\
            utils.get_model_from_pretrain(model_path=model_path,
                                          config=config,
                                          resume=resume)

        model.cuda(0)

        criterion = getattr(losses, config['criterion'])(**config['criterion_args'])

        optimizer = getattr(torch.optim, config['optimizer'])(
            model.parameters(), **config['optimizer_args'])

        scheduler = getattr(schedulers, config['scheduler'])(
            optimizer, **config['scheduler_args'])
        if optimizer_params and resume:
            optimizer.load_state_dict(optimizer_params)
        if scheduler_params and resume:
            scheduler.load_state_dict(scheduler_params)


        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                model_out = Runner._forward_audio(
                    model, batch, device=0)
                loss = criterion(model_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                model_out = Runner._forward_audio(
                    model, batch, device=0)
                loss = criterion(model_out)

                logits = model_out['logit'].cpu()
                targets = model_out['target'].cpu()
            return loss.item(), logits, targets
        
        trainer, evaluator = Engine(_train), Engine(_inference)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        Average(output_transform=lambda x: x[0]).attach(evaluator, 'Loss')
        Accuracy(output_transform=lambda x: (x[1], x[2])).attach(evaluator, 'Acc')

        ProgressBar(persist=False, ncols=75).attach(
            trainer, output_transform=lambda x: {'loss': x})
        ProgressBar(persist=False, ncols=75, desc='Evaluating').attach(
            evaluator, output_transform=None)
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate(engine):
            
            lr = optimizer.param_groups[0]['lr']
            lr = round(lr, 7)
            logger.info(f'<==== Epoch {trainer.state.epoch}, lr {lr} ====>')
            global_train_loss = engine.state.metrics['Loss']
            logger.info('Training Loss: {:<5.2f}'.format(global_train_loss))

            evaluator.run(ValDataloader, max_epochs=1)
            val_loss = evaluator.state.metrics['Loss']
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            logger.info('Validation Loss: {:<5.2f}'.format(val_loss))
            acc = evaluator.state.metrics['Acc']
            logger.info('Validation Acc: {:<5.2f}'.format(acc))
        

        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, EarlyStoppingHandler)

        # BestModelCheckpoint = ModelCheckpoint(
        #     outputdir, filename_prefix='train_best',
        #     score_function=lambda engine: -engine.state.metrics['Loss'],
        #     score_name='Loss', n_saved=1,
        #     global_step_transform=global_step_from_engine(trainer))
        # Save model which has the best training loss

        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='eval_best',
            score_function=lambda engine: engine.state.metrics['Acc'],
            score_name='Acc', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        # Save model which has the best validation accuracy

        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, 
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
        )

        # trainer.add_event_handler(
        #     Events.EPOCH_COMPLETED, BestModelCheckpoint, 
        #     {
        #         'model': model,
        #         'optimizer': optimizer,
        #         'scheduler': scheduler
        #     }
        # )
        # Save model which has the best training loss

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, BestModelCheckpoint, 
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
        )
        # Save model which has the best validation accuracy

        trainer.run(TrainDataloader, max_epochs=config['n_epochs'])

        return outputdir

    def test(self, **kwargs):
        print(kwargs)
if __name__ == '__main__':
    fire.Fire(Runner)

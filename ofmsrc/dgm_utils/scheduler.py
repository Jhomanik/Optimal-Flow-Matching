import wandb
import os

# for FID, IS calculation
# from pytorch_gan_metrics import get_inception_score_and_fid

# free memory
import gc
import torch


class TrainingSchedulerGeneric:

    @staticmethod
    def extract_kwargs(kwargs, *names, del_names=True):
        vals = []
        for name in names:
            vals.append(kwargs[name])
            if del_names:
                del kwargs[name]
        if len(vals) == 1:
            return vals[0]
        return tuple(vals)

    def __init__(self, *args, **kwargs):
        self._steps_counter = 0

    def on_batch_optim_step(self, epoch=None, batch=None):
        pass

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        self._steps_counter += 1

    def on_epoch_train_end(self, epoch=None):
        gc.collect(); torch.cuda.empty_cache()

    def on_epoch_eval_end(self, epoch=None, losses=None):
        gc.collect(); torch.cuda.empty_cache()


class TrainingSchedulerSM_Mixin(TrainingSchedulerGeneric):

    def __init__(self, *args, **kwargs):
        self.SMscheduler = self.extract_kwargs(kwargs, 'SMscheduler')
        super().__init__(*args, **kwargs)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        super().on_batch_train_end(epoch, batch, losses, data)
        for k, v in losses.items():
            self.SMscheduler.SM.upd("train_{}".format(k), v.item())

    def on_epoch_eval_end(self, epoch=None, losses=None):
        super().on_epoch_eval_end(epoch, losses)
        for k, v in losses.items():
            self.SMscheduler.SM.upd("test_{}".format(k), v)
        self.SMscheduler.epoch()


class TrainingSchedulerWandB_Mixin(TrainingSchedulerGeneric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = self.extract_kwargs(kwargs, 'use_wandb', del_names=False)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        super().on_batch_train_end(epoch, batch, losses, data)
        if self.use_wandb:
            res_dict = {'epoch': epoch}
            for k, v in losses.items():
                res_dict[k] = v
            wandb.log({'train': res_dict}, step=self._steps_counter)

    def on_epoch_eval_end(self, epoch=None, losses=None):
        super().on_epoch_eval_end(epoch, losses)
        if self.use_wandb:
            res_dict = {}
            for k, v in losses.items():
                res_dict[k] = v
            wandb.log({'test': res_dict}, step=self._steps_counter)
        else:
            pass #TODO: add simple print of values


class TrainingSchedulerModelsSaver_Mixin(TrainingSchedulerGeneric):

    def __init__(
        self,
        *args,
        save_models_interval=100,
        rewrite_saved_models=True,
        **kwargs
    ):
        self.model = self.extract_kwargs(
            kwargs, 'model', del_names=False)
        self.save_models_path = self.extract_kwargs(
            kwargs, 'save_models_path')
        self.save_models_interval = save_models_interval
        self.rewrite_saved_models = rewrite_saved_models
        super().__init__(*args, **kwargs)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        if self._steps_counter % self.save_models_interval == 0:
            self.model.store(os.path.join(self.save_models_path, 'model_latest.pth'))
            if not self.rewrite_saved_models:
                file_name = "model_step_{}.pth".format(self._steps_counter)
                self.model.store(os.path.join(self.save_models_path, file_name))
        super().on_batch_train_end(epoch, batch, losses, data)

class TrainingSchedulerFID_IS_Mixin(TrainingSchedulerGeneric):

    def sample_from_model(self, n_samples):
        raise NotImplementedError()

    def __init__(
        self,
        *args,
        save_fid_is_interval=100,
        inception_device='cuda:0',
        inception_batch_size=64,
        estimate_fid_is_n_samples=1000,
        **kwargs
    ):
        self.reference_inception_features_path = self.extract_kwargs(
            kwargs, 'reference_inception_features_path')
        self.compute_fid_is = self.extract_kwargs(kwargs, 'compute_fid_is', del_names=False)
        self.save_fid_is_interval = save_fid_is_interval
        self.inception_device = inception_device
        self.inception_batch_size = inception_batch_size
        self.estimate_fid_is_n_samples = estimate_fid_is_n_samples
        super().__init__(*args, **kwargs)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        if self.compute_fid_is:
            if self._steps_counter % self.save_fid_is_interval == 0:
                ims = self.sample_from_model(self.estimate_fid_is_n_samples)

                (IS_score, IS_score_std), fid_score = get_inception_score_and_fid(
                    ims, self.reference_inception_features_path, 
                    device=self.inception_device, use_torch=False, 
                    batch_size=self.inception_batch_size
                )
                # gc.collect(); torch.cuda.empty_cache()
                losses['IS_score'] = IS_score
                losses['IS_score_std'] = IS_score_std
                losses['FID_score'] = fid_score
                gc.collect(); torch.cuda.empty_cache()

        super().on_batch_train_end(epoch, batch, losses, data)

class TrainingSchedulerLR_Mixin(TrainingSchedulerGeneric):

    def __init__(self, *args, **kwargs):
        self.lr_scheduler = self.extract_kwargs(
            kwargs, 'lr_scheduler', del_names=True)
        super().__init__(*args, **kwargs)

    def on_epoch_eval_end(self, epoch=None, losses=None):
        losses['lr'] = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()
        super().on_epoch_eval_end(epoch, losses)


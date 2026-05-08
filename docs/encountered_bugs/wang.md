# Bugs encountered during Openfold finetuning

### Prerequisites
- Inputted soloseq MSA output into funetuning. But the formats are different, so used an AI written functions, which should be incorrect, to parse the contents in xxx.pt files.

### Finetuing command
\$ python train_openfold.py data/pdb_recent/selected_sequences/ data/pdb_recent/embeddings_output_dir/ data/pdb_recent/mmcif_files/ data/pdb_recent/soloseq_finetuning_outputs 2026-01-18 --use_single_seq_mode True --config_preset seq_model_esm1b_ptm --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm_finetuning_260331.pt --resume_model_weights_only True --template_release_dates_cache_path data/pdb_recent/solo_mmcif_cache.json --train_chain_data_cache_path data/pdb_recent/solo_chain_data_cache.json --precision bf16-mixed --gpus 2 --seed 42 --deepspeed_config_path deepspeed_config.json

### Errors
/export/home/j-wang/CUDA_Projects/Fed-fold/train_openfold.py:310: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  sd = torch.load(args.resume_from_ckpt)
WARNING:root:Removing 3212 alignment entries (9M68_G, 9KBK_B, 9PF1_F, 9BFC_F, 9P9J_La, 9P9J_LA, 9L1H_A, 9BHH_X, 9UTK_A, 9MKN_H, ...) with no corresponding entries in chain_data_cache (data/pdb_recent/solo_chain_data_cache.json).
initializing deepspeed distributed: GLOBAL_RANK: 1, MEMBER: 2/2
WARNING:root:Removing 3212 alignment entries (9M68_G, 9KBK_B, 9PF1_F, 9BFC_F, 9P9J_La, 9P9J_LA, 9L1H_A, 9BHH_X, 9UTK_A, 9MKN_H, ...) with no corresponding entries in chain_data_cache (data/pdb_recent/solo_chain_data_cache.json).
WARNING:root:Removing 3212 alignment entries (9M68_G, 9KBK_B, 9PF1_F, 9BFC_F, 9P9J_La, 9P9J_LA, 9L1H_A, 9BHH_X, 9UTK_A, 9MKN_H, ...) with no corresponding entries in chain_data_cache (data/pdb_recent/solo_chain_data_cache.json).
[rank1]: Traceback (most recent call last):
[rank1]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/train_openfold.py", line 703, in <module>
[rank1]:     main(args)
[rank1]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/train_openfold.py", line 452, in main
[rank1]:     trainer.fit(
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
[rank1]:     call._call_and_handle_interrupt(
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 48, in _call_and_handle_interrupt
[rank1]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
[rank1]:     return function(*args, **kwargs)
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
[rank1]:     self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1053, in _run
[rank1]:     self.strategy.setup(self)
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 359, in setup
[rank1]:     self._init_config_if_needed()
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 828, in _init_config_if_needed
[rank1]:     self._format_config()
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 837, in _format_config
[rank1]:     self._format_batch_size_and_grad_accum_config()
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 930, in _format_batch_size_and_grad_accum_config
[rank1]:     batch_size = self._auto_select_batch_size()
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 942, in _auto_select_batch_size
[rank1]:     train_dataloader = data_source.dataloader()
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py", line 302, in dataloader
[rank1]:     return call._call_lightning_datamodule_hook(self.instance.trainer, self.name)
[rank1]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 199, in _call_lightning_datamodule_hook
[rank1]:     return fn(*args, **kwargs)
[rank1]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 1084, in train_dataloader
[rank1]:     return self._gen_dataloader("train")
[rank1]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 1061, in _gen_dataloader
[rank1]:     dataset.reroll()
[rank1]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 693, in reroll
[rank1]:     datapoint_idx = next(samples)
[rank1]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 652, in looped_samples
[rank1]:     candidate_idx = next(idx_iter)
[rank1]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 634, in looped_shuffled_dataset_idx
[rank1]:     shuf = torch.multinomial(
[rank1]: RuntimeError: cannot sample n_sample <= 0 samples
[rank0]: Traceback (most recent call last):
[rank0]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/train_openfold.py", line 703, in <module>
[rank0]:     main(args)
[rank0]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/train_openfold.py", line 452, in main
[rank0]:     trainer.fit(
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
[rank0]:     call._call_and_handle_interrupt(
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 48, in _call_and_handle_interrupt
[rank0]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
[rank0]:     return function(*args, **kwargs)
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
[rank0]:     self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1053, in _run
[rank0]:     self.strategy.setup(self)
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 359, in setup
[rank0]:     self._init_config_if_needed()
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 828, in _init_config_if_needed
[rank0]:     self._format_config()
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 837, in _format_config
[rank0]:     self._format_batch_size_and_grad_accum_config()
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 930, in _format_batch_size_and_grad_accum_config
[rank0]:     batch_size = self._auto_select_batch_size()
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/strategies/deepspeed.py", line 942, in _auto_select_batch_size
[rank0]:     train_dataloader = data_source.dataloader()
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py", line 302, in dataloader
[rank0]:     return call._call_lightning_datamodule_hook(self.instance.trainer, self.name)
[rank0]:   File "/home/j-wang/miniconda3/envs/openfold_env/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 199, in _call_lightning_datamodule_hook
[rank0]:     return fn(*args, **kwargs)
[rank0]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 1084, in train_dataloader
[rank0]:     return self._gen_dataloader("train")
[rank0]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 1061, in _gen_dataloader
[rank0]:     dataset.reroll()
[rank0]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 693, in reroll
[rank0]:     datapoint_idx = next(samples)
[rank0]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 652, in looped_samples
[rank0]:     candidate_idx = next(idx_iter)
[rank0]:   File "/export/home/j-wang/CUDA_Projects/Fed-fold/openfold/data/data_modules.py", line 634, in looped_shuffled_dataset_idx
[rank0]:     shuf = torch.multinomial(
[rank0]: RuntimeError: cannot sample n_sample <= 0 samples
[rank0]:[W508 16:09:46.186365765 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
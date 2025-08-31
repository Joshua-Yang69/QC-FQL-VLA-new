
if step != 0 and step % self.save_every_n_steps == 0:
                    self.accelerator.wait_for_everyone()
                    self.store_model_weights(self.working_dir, f"{self.global_step}_")
                    if self.accelerator.is_main_process:
                        log.info(f'{self.global_step} Steps reached. Stored weights updated!')
                if step % 5000 == 0 and hasattr(self.train_dataloader.dataset, 'tracker') and getattr(self.train_dataloader.dataset.tracker, 'enable_tracking', False):
                    self.train_dataloader.dataset.tracker.replace_overused_samples()


        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")



    def store_model_weights(self, store_path: str, additional_name: str = ""):
        """
        Saves model weights and EMA weights separately.
        """
        try:
            # Ensure the directory exists
            os.makedirs(store_path, exist_ok=True)

            # Synchronize all processes
            self.accelerator.wait_for_everyone()

            # Only save from main process to avoid conflicts
            if self.accelerator.is_main_process:
                # Get unwrapped model
                model = self.accelerator.unwrap_model(self.agent)

                # Save default model weights to CPU
                default_weights_path = os.path.join(
                    store_path, f"{self.global_step}_default_weights.pt"
                )
                model_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
                torch.save(model_state, default_weights_path)

                # Save EMA weights if available
                if self.use_ema and self.ema is not None:
                    ema_weights_path = os.path.join(
                        store_path, f"{self.global_step}_ema_weights.pt"
                    )
                    ema_model = self.accelerator.unwrap_model(self.ema)
                    ema_state = {
                        k: v.detach().cpu() for k, v in ema_model.state_dict().items()
                    }
                    torch.save(ema_state, ema_weights_path)

            # Synchronize again
            self.accelerator.wait_for_everyone()

            # Let Accelerate handle full checkpoint
            full_state_path = os.path.join(store_path, f"checkpoint_{self.global_step}")
            self.accelerator.save_state(full_state_path)

        except Exception as e:
            log.error(f"Error in store_model_weights: {e}")
            import traceback
            traceback.print_exc()

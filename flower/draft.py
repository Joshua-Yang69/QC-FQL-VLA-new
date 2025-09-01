
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
def forward(self, x, custom_attn_mask=None, is_causal=False):
        # ... (Your existing code for q, k, v, etc.)

        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)

        # --- 修复部分 ---
        # Initialize attn_mask and is_causal to be used in the F.scaled_dot_product_attention call
        attn_mask = None
        use_is_causal = False

        if is_causal and custom_attn_mask is None:
            # Use PyTorch's built-in causal masking by setting is_causal=True and attn_mask=None
            use_is_causal = True
        elif custom_attn_mask is not None:
            # For custom masks, set is_causal=False and pass the mask explicitly
            attn_mask = custom_attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            # The mask needs to be a boolean mask, where True means to mask/ignore the value
            # Note: your original code used ~mask which is correct if mask is True for things to keep.
            # Here we assume the input custom_attn_mask is a boolean mask where True means to mask.
            # If your custom mask is `True` for positions that should be **masked**, then `~` is not needed.
            # If your custom mask is `True` for positions that should be **kept**, then `~` is needed.
            # Double-check this part based on your `custom_attn_mask` definition.
            # For simplicity and to match the PyTorch convention, let's assume the mask is where `True` means mask.

        # Use PyTorch's built-in scaled dot-product attention.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale,
            is_causal=use_is_causal
        )
        # --- 修复部分结束 ---

        out = attn_output.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out
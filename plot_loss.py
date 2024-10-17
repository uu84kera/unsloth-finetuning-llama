import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_losses(log_dir, save_dir=None):
    # Iterate over all event files in the log directory
    for root, _, files in os.walk(log_dir):
        for i, file in enumerate(files):
            if file.startswith("events.out.tfevents"):
                event_path = os.path.join(root, file)
                # Load TensorBoard logs
                event_acc = EventAccumulator(event_path)
                event_acc.Reload()

                # Get train and eval loss tags
                tags = event_acc.Tags().get('scalars', [])
                train_loss_tag = 'train/loss'
                eval_loss_tag = 'eval/loss'

                train_steps, train_losses = [], []
                eval_steps, eval_losses = [], []

                if train_loss_tag in tags:
                    # Extract train loss
                    train_loss_events = event_acc.Scalars(train_loss_tag)
                    train_steps = [e.step for e in train_loss_events]
                    train_losses = [e.value for e in train_loss_events]

                if eval_loss_tag in tags:
                    # Extract eval loss
                    eval_loss_events = event_acc.Scalars(eval_loss_tag)
                    eval_steps = [e.step for e in eval_loss_events]
                    eval_losses = [e.value for e in eval_loss_events]

                # Create save directory if specified
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

                # Plot train vs eval loss on the same plot
                plt.figure(figsize=(10, 5))
                if train_steps and train_losses:
                    plt.plot(train_steps, train_losses, label='Train Loss', color='b')
                if eval_steps and eval_losses:
                    plt.plot(eval_steps, eval_losses, label='Eval Loss', color='r')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Train vs Eval Loss for Llama')
                plt.legend()
                plt.grid(True)
                if save_dir:
                    combined_loss_path = os.path.join(save_dir, f"train_eval_loss_plot_file_{i+1}.png")
                    plt.savefig(combined_loss_path)
                    print(f"Train vs Eval loss plot saved at {combined_loss_path}")
                else:
                    plt.show()

if __name__ == "__main__":
    log_directory = "logs"  # Replace with the actual log directory used during training
    save_directory = "plot"  # Specify the directory to save the plots, or set to None to show the plots
    plot_losses(log_directory, save_directory)

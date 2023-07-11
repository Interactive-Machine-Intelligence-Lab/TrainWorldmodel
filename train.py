import pickle
from tqdm import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt
from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig
from models.world_model import WorldModel, TransformerConfig
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp

from utils import compute_loss
class Trainer:
    def __init__(self, vocab_size, rank, world_size, dir_name) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.vocab_size = vocab_size

        self.dir_name = dir_name

        self.batch_size = 32
        self.epochs = 100000
        self.learning_rate = 0.0001
        self.max_grad_norm = 10.0
        self.test_num = 32
        self.scaler = torch.cuda.amp.GradScaler()
        self.train_size = 26000
        self.num_gpus = torch.cuda.device_count()
        self.setup(rank, world_size)

    def setup(self, rank, world_size):
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=world_size,
            rank=rank)
        
    def cleanup(self):
        dist.destroy_process_group()

    def load_dataset(self, train_size):
        obs = np.load("X_obs_resized.npy")
        obs = torch.from_numpy(obs).permute(0, 1, 4, 2, 3)

        act = np.load("X_act.npy")
        print(obs.shape)
        print(act.shape)

        obs_test = obs[train_size:]
        act_test = act[train_size:]

        obs_train = obs[:train_size]
        act_train = act[:train_size]

        train_dataset = TensorDataset(torch.tensor(obs_train), torch.tensor(act_train))
        test_dataset = TensorDataset(torch.tensor(obs_test), torch.tensor(act_test))

        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank)

        train_loader = DataLoader(train_dataset, sampler=train_sampler, pin_memory=True, num_workers=4 * self.num_gpus, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, pin_memory=True, num_workers=4 * self.num_gpus, batch_size=self.batch_size)

        return train_loader, test_loader

    def build_tokenizer(self):
        config = EncoderDecoderConfig(
            resolution=256,
            in_channels=3,
            z_channels=256,
            ch=128,
            ch_mult=[1, 1, 1, 2, 2, 4],
            num_res_blocks=2,
            out_ch=3,
            dropout=0.0,
            attn_resolutions=[16],
        )
        encoder = Encoder(config)
        decoder = Decoder(config)
        tokenizer = Tokenizer(
            vocab_size=self.vocab_size, embed_dim=1024, encoder=encoder, decoder=decoder
        )

        tokenizer_state_dict = torch.load(
            "checkpoint/tokenizer.pt", map_location=self.device
        )
        tokenizer.load_state_dict(tokenizer_state_dict)
        tokenizer.eval()
        return tokenizer

    def build_worldmodel(self):
        t = TransformerConfig(
            tokens_per_block=17,
            max_blocks=20,
            attention="causal",
            num_layers=10,
            num_heads=4,
            embed_dim=256,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        vocab_size = 512

        world_model = WorldModel(vocab_size, 7, t)
        world_model.train()
        return world_model

    def run(self):
        train_loader, test_loader = self.load_dataset(self.train_size)
        train_losses = []
        test_losses = []
        

        mask_fill = torch.logical_not(torch.cat((torch.ones(self.batch_size, 20)), dim=-1,)).to(self.device, non_blocking=True)
        tokenizer = self.build_tokenizer().to(self.device)
        tokenizer = torch.compile(tokenizer)
        worldmodel = self.build_worldmodel().to(self.device)
        worldmodel = DDP(worldmodel, device_ids=[self.rank])
        worldmodel = torch.compile(worldmodel)

        optimizer = torch.optim.Adam(worldmodel.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            print(epoch, "Training.")
            loss_total_epoch = []
            test_error_total_epoch = []

            tqdm_leave = False
            if self.rank == 1:
                tqdm_leave = True

            for x, x_act in tqdm(train_loader, position=self.rank, leave=tqdm_leave, desc=f"{epoch} Training (Rank : {self.rank})"):
                optimizer.zero_grad(set_to_none=True)
                x = x.to(self.device, non_blocking=True).float() / 255.
                x_act = x_act.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    losses = compute_loss(worldmodel, x, x_act, mask_fill, tokenizer)
                loss_total_step = losses
                self.scaler.scale(loss_total_step).backward()
                loss_total_epoch.append(loss_total_step.item())

                torch.nn.utils.clip_grad_norm_(worldmodel.parameters(), self.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            epoch_train_loss = np.mean(loss_total_epoch)
            train_losses.append(epoch_train_loss)
            # Wait
            torch.distributed.barrier()
            print("############## Epoch: ", epoch, ", Train loss: ", epoch_train_loss, ", Test error: ", epoch_test_error, " ##############")
            for x, x_act in test_loader:
                x = x.to(self.device, non_blocking=True).float() / 255.
                x_act = x_act.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(), torch.no_grad():
                    losses = compute_loss(x, x_act, mask_fill, tokenizer)
                    test_error_total_epoch.append(losses.item())
                epoch_test_error = np.mean(test_error_total_epoch)
                test_losses.append(epoch_test_error)
            print("############## Epoch: ", epoch, ", Test loss: ", epoch_test_error, ", Test error: ", epoch_test_error, " ##############")
            torch.distributed.barrier()

            if epoch % 100 == 0 and self.rank == 0:
                plt.close()
                plt.plot(train_losses)
                plt.savefig("{}/train_losses.png".format(self.dir_name))
                plt.close()
                plt.plot(test_losses)
                plt.savefig("{}/test_losses.png".format(self.dir_name))

                torch.save(worldmodel.state_dict(), "{}/world_model.pt".format(self.dir_name))
                pickle.dump(train_losses, open("{}/train_losses.pkl".format(self.dir_name), "wb"),)

        self.cleanup()


def main_worker(gpu, ngpus_per_node, vocab_size, dir_name):
    trainer = Trainer(vocab_size=vocab_size, rank=gpu, world_size=ngpus_per_node, dir_name=dir_name)
    trainer.run()

def main():
    vocab_size = 512
    dir_name = "output"
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, vocab_size, dir_name))


if __name__ == "__main__":
    main()

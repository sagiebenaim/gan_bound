from itertools import chain

import torch.optim as optim
from torch.autograd import Variable
from progressbar import ETA, Bar, Percentage, ProgressBar

from dataset import *
from disco_gan_model import DiscoGAN
from model import *

class GeneralGANBound(DiscoGAN):

    def to_no_grad_var(self, var):
        data = self.as_np(var)
        var = Variable(torch.FloatTensor(data), requires_grad=False)
        if self.cuda:
            var = var.cuda()
        return var

    def initialize(self):

        self.cuda = self.args.cuda
        if self.cuda == 'true':
            self.cuda = True
        else:
            self.cuda = False

        self.result_path = os.path.join(self.args.result_path, self.args.task_name)
        if self.args.style_A:
            self.result_path = os.path.join(self.result_path, self.args.style_A)
        self.result_path = os.path.join(self.result_path, self.args.model_arch)

        self.model_path = os.path.join(self.args.model_path, self.args.task_name)
        if self.args.style_A:
            self.model_path = os.path.join(self.model_path, self.args.style_A)
        self.model_path = os.path.join(self.model_path, self.args.model_arch)

        number_of_samples = self.args.number_of_samples
        if not self.args.not_all_samples:
            number_of_samples = None

        self.data_style_A, self.data_style_B, self.test_style_A, self.test_style_B = self.get_data(number_of_samples=number_of_samples)
        self.alligned_data_style_A, self.alligned_data_style_B, self.alligned_test_style_A, self.alligned_test_style_B = self.get_data(number_of_samples=number_of_samples)

        self.test_A, self.test_B = self.get_images(self.test_style_A, self.test_style_B)
        self.test_A = Variable(torch.FloatTensor(self.test_A), volatile=True)
        self.test_B = Variable(torch.FloatTensor(self.test_B), volatile=True)

        generator_extension_folders = ["Generator_1", "Generator_2"]
        self.result_paths = []
        self.model_paths =[]
        for extension_folder in generator_extension_folders:
            result_path = os.path.join(self.result_path, extension_folder)
            model_path = os.path.join(self.model_path, extension_folder)
            self.result_paths.append(result_path)
            self.model_paths.append(model_path)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

        self.generator_A_1 = Generator(num_layers=self.args.num_layers)
        self.generator_B_1 = Generator(num_layers=self.args.num_layers)
        self.discriminator_A_1 = Discriminator()
        self.discriminator_B_1 = Discriminator()

        self.generator_A_2 = Generator(num_layers=self.args.num_layers)
        self.generator_B_2 = Generator(num_layers=self.args.num_layers)
        self.discriminator_A_2 = Discriminator()
        self.discriminator_B_2 = Discriminator()

        if self.cuda:
            self.test_A = self.test_A.cuda()
            self.test_B = self.test_B.cuda()

            self.generator_A_1 = self.generator_A_1.cuda()
            self.generator_B_1 = self.generator_B_1.cuda()
            self.discriminator_A_1 = self.discriminator_A_1.cuda()
            self.discriminator_B_1 = self.discriminator_B_1.cuda()

            self.generator_A_2 = self.generator_A_2.cuda()
            self.generator_B_2 = self.generator_B_2.cuda()
            self.discriminator_A_2 = self.discriminator_A_2.cuda()
            self.discriminator_B_2 = self.discriminator_B_2.cuda()

        data_size = min(len(self.data_style_A), len(self.data_style_B))
        self.n_batches = (data_size // self.args.batch_size)

        self.recon_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCELoss()
        self.feat_criterion = nn.HingeEmbeddingLoss()
        self.correlation_criterion = nn.L1Loss()
        self.ground_truth_critertion = nn.L1Loss()

        gen_params_1 = chain(self.generator_A_1.parameters(), self.generator_B_1.parameters())
        dis_params_1 = chain(self.discriminator_A_1.parameters(), self.discriminator_B_1.parameters())
        gen_params_2 = chain(self.generator_A_2.parameters(), self.generator_B_2.parameters())
        dis_params_2 = chain(self.discriminator_A_2.parameters(), self.discriminator_B_2.parameters())

        self.optim_gen_1 = optim.Adam(gen_params_1, lr=self.args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optim_dis_1 = optim.Adam(dis_params_1, lr=self.args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optim_gen_2 = optim.Adam(gen_params_2, lr=self.args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
        self.optim_dis_2 = optim.Adam(dis_params_2, lr=self.args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

    def calculate_losses(self, A, B, fixed_batch_A=None, fixed_batch_B=None, fixed_sample_index=None):

        AB_1 = self.generator_B_1(A)
        BA_1 = self.generator_A_1(B)

        ABA_1 = self.generator_A_1(AB_1)
        BAB_1 = self.generator_B_1(BA_1)

        AB_2 = self.generator_B_2(A)
        BA_2 = self.generator_A_2(B)

        ABA_2 = self.generator_A_2(AB_2)
        BAB_2 = self.generator_B_2(BA_2)

        # Reconstruction Loss
        recon_loss_A_1 = self.recon_criterion(ABA_1, A)
        recon_loss_B_1 = self.recon_criterion(BAB_1, B)

        recon_loss_A_2 = self.recon_criterion(ABA_2, A)
        recon_loss_B_2 = self.recon_criterion(BAB_2, B)

        # Real/Fake GAN Loss (A)
        A_dis_real_1, A_feats_real_1 = self.discriminator_A_1(A)
        A_dis_fake_1, A_feats_fake_1 = self.discriminator_A_1(BA_1)

        dis_loss_A_1, gen_loss_A_1 = self.get_gan_loss(A_dis_real_1, A_dis_fake_1)
        fm_loss_A_1 = self.get_fm_loss(A_feats_real_1, A_feats_fake_1)

        A_dis_real_2, A_feats_real_2 = self.discriminator_A_2(A)
        A_dis_fake_2, A_feats_fake_2 = self.discriminator_A_2(BA_2)

        dis_loss_A_2, gen_loss_A_2 = self.get_gan_loss(A_dis_real_2, A_dis_fake_2)
        fm_loss_A_2 = self.get_fm_loss(A_feats_real_2, A_feats_fake_2)

        # Real/Fake GAN Loss (B)
        B_dis_real_1, B_feats_real_1 = self.discriminator_B_1(B)
        B_dis_fake_1, B_feats_fake_1 = self.discriminator_B_1(AB_1)

        dis_loss_B_1, gen_loss_B_1 = self.get_gan_loss(B_dis_real_1, B_dis_fake_1)
        fm_loss_B_1 = self.get_fm_loss(B_feats_real_1, B_feats_fake_1)

        B_dis_real_2, B_feats_real_2 = self.discriminator_B_2(B)
        B_dis_fake_2, B_feats_fake_2 = self.discriminator_B_2(AB_2)

        dis_loss_B_2, gen_loss_B_2 = self.get_gan_loss(B_dis_real_2, B_dis_fake_2)
        fm_loss_B_2 = self.get_fm_loss(B_feats_real_2, B_feats_fake_2)

        # Correlation loss
        # Distance between generator 1 and generator 2's output
        correlation_loss_AB_2 = - self.correlation_criterion(AB_2, self.to_no_grad_var(AB_1))
        correlation_loss_BA_2 = - self.correlation_criterion(BA_2, self.to_no_grad_var(BA_1))

        # Total Loss
        if self.iters < self.args.gan_curriculum:
            rate = self.args.starting_rate
            correlation_rate = self.args.starting_correlation_rate
        else:
            rate = self.args.default_rate
            correlation_rate = self.args.default_correlation_rate

        gen_loss_A_1_total = (gen_loss_B_1 * 0.1 + fm_loss_B_1 * 0.9) * (1. - rate) + recon_loss_A_1 * rate
        gen_loss_B_1_total = (gen_loss_A_1 * 0.1 + fm_loss_A_1 * 0.9) * (1. - rate) + recon_loss_B_1 * rate

        gen_loss_A_2_total = (gen_loss_B_2 * 0.1 + fm_loss_B_2 * 0.9) * (1. - rate) + recon_loss_A_2 * rate + correlation_loss_AB_2 * correlation_rate
        gen_loss_B_2_total = (gen_loss_A_2 * 0.1 + fm_loss_A_2 * 0.9) * (1. - rate) + recon_loss_B_2 * rate + correlation_loss_BA_2 * correlation_rate

        if self.args.model_arch == 'discogan':
            gen_loss_1 = gen_loss_A_1_total + gen_loss_B_1_total
            dis_loss_1 = dis_loss_A_1 + dis_loss_B_1
            gen_loss_2 = gen_loss_A_2_total + gen_loss_B_2_total
            dis_loss_2 = dis_loss_A_2 + dis_loss_B_2

        return gen_loss_1, dis_loss_1, gen_loss_2, dis_loss_2, \
               - correlation_loss_AB_2, - correlation_loss_BA_2,

    def run(self):

        self.initialize()
        self.iters = 0

        current_correlation_loss_AB = 0
        current_correlation_loss_BA = 0

        for epoch in range(self.args.epoch_size):
            data_style_A, data_style_B = shuffle_data(self.data_style_A, self.data_style_B)

            widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=self.n_batches, widgets=widgets)
            pbar.start()

            current_correlation_loss_AB = 0
            current_correlation_loss_BA = 0

            for i in range(self.n_batches):

                pbar.update(i)

                self.generator_A_1.zero_grad()
                self.generator_B_1.zero_grad()
                self.discriminator_A_1.zero_grad()
                self.discriminator_B_1.zero_grad()

                self.generator_A_2.zero_grad()
                self.generator_B_2.zero_grad()
                self.discriminator_A_2.zero_grad()
                self.discriminator_B_2.zero_grad()

                A_path = data_style_A[i * self.args.batch_size: (i + 1) * self.args.batch_size]
                B_path = data_style_B[i * self.args.batch_size: (i + 1) * self.args.batch_size]

                A, B = self.get_images(A_path, B_path)
                A = Variable(torch.FloatTensor(A))
                B = Variable(torch.FloatTensor(B))
                if self.cuda:
                    A = A.cuda()
                    B = B.cuda()

                self.gen_loss_1, self.dis_loss_1, self.gen_loss_2, self.dis_loss_2, correlation_loss_AB, correlation_loss_BA = self.calculate_losses(A, B)

                current_correlation_loss_AB += correlation_loss_AB
                current_correlation_loss_BA += correlation_loss_BA

                self.finish_iteration()
                self.iters += 1

        return current_correlation_loss_AB/self.n_batches, current_correlation_loss_BA/self.n_batches

    def finish_iteration(self):

        if self.iters % self.args.update_interval == 0:
            self.dis_loss_1.backward()
            self.optim_dis_1.step()
            self.dis_loss_2.backward()
            self.optim_dis_2.step()
        else:
            self.gen_loss_1.backward()
            self.optim_gen_1.step()
            self.gen_loss_2.backward()
            self.optim_gen_2.step()

        if self.iters % self.args.image_save_interval == 0:

            AB_1 = self.generator_B_1(self.test_A)
            BA_1 = self.generator_A_1(self.test_B)
            ABA_1 = self.generator_A_1(AB_1)
            BAB_1 = self.generator_B_1(BA_1)

            AB_2 = self.generator_B_2(self.test_A)
            BA_2 = self.generator_A_2(self.test_B)
            ABA_2 = self.generator_A_2(AB_2)
            BAB_2 = self.generator_B_2(BA_2)

            n_testset = min(self.test_A.size()[0], self.test_B.size()[0])
            subdir_path_1 = os.path.join(self.result_paths[0], str(self.iters / self.args.image_save_interval))
            subdir_path_2 = os.path.join(self.result_paths[1], str(self.iters / self.args.image_save_interval))

            if not os.path.exists(subdir_path_1):
                os.makedirs(subdir_path_1)
            if not os.path.exists(subdir_path_2):
                os.makedirs(subdir_path_2)

            for im_idx in range(n_testset):
                A_val = self.test_A[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                B_val = self.test_B[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                BA_1_val = BA_1[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                ABA_1_val = ABA_1[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                AB_1_val = AB_1[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                BAB_1_val = BAB_1[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.

                BA_2_val = BA_2[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                ABA_2_val = ABA_2[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                AB_2_val = AB_2[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                BAB_2_val = BAB_2[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.

                filename_prefix_1 = os.path.join(subdir_path_1, str(im_idx))
                filename_prefix_2 = os.path.join(subdir_path_2, str(im_idx))

                scipy.misc.imsave(filename_prefix_1 + '.A.jpg', A_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_1 + '.B.jpg', B_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_1 + '.BA.jpg', BA_1_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_1 + '.AB.jpg', AB_1_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_1 + '.ABA.jpg', ABA_1_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_1 + '.BAB.jpg', BAB_1_val.astype(np.uint8)[:, :, ::-1])

                scipy.misc.imsave(filename_prefix_2 + '.A.jpg', A_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_2 + '.B.jpg', B_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_2 + '.BA.jpg', BA_2_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_2 + '.AB.jpg', AB_2_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_2 + '.ABA.jpg', ABA_2_val.astype(np.uint8)[:, :, ::-1])
                scipy.misc.imsave(filename_prefix_2 + '.BAB.jpg', BAB_2_val.astype(np.uint8)[:, :, ::-1])

        if self.iters % self.args.model_save_interval == 0:
            torch.save(self.generator_A_1,
                       os.path.join(self.model_paths[0], 'model_gen_A-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.generator_B_1,
                       os.path.join(self.model_paths[0], 'model_gen_B-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.discriminator_A_1,
                       os.path.join(self.model_paths[0], 'model_dis_A-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.discriminator_B_1,
                       os.path.join(self.model_paths[0], 'model_dis_B-' + str(self.iters / self.args.model_save_interval)))

            torch.save(self.generator_A_2,
                       os.path.join(self.model_paths[1], 'model_gen_A-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.generator_B_2,
                       os.path.join(self.model_paths[1], 'model_gen_B-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.discriminator_A_2,
                       os.path.join(self.model_paths[1], 'model_dis_A-' + str(self.iters / self.args.model_save_interval)))
            torch.save(self.discriminator_B_2,
                       os.path.join(self.model_paths[1], 'model_dis_B-' + str(self.iters / self.args.model_save_interval)))



if __name__ == "__main__":
    model = GeneralGANBound()
    model.run()
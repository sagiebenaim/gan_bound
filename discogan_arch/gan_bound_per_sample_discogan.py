from progressbar import ETA, Bar, Percentage, ProgressBar
from torch.autograd import Variable

from dataset import *
from general_gan_bound_discogan import GeneralGANBound
from model import *

class GanBoundPerSample(GeneralGANBound):

    def calculate_losses(self, A, B, fixed_batch_A=None, fixed_batch_B=None, fixed_sample_index=None):

        AB_1 = self.generator_B_1(A)
        BA_1 = self.generator_A_1(B)

        ABA_1 = self.generator_A_1(AB_1)
        BAB_1 = self.generator_B_1(BA_1)

        fixed_batch_AB_1 = self.generator_B_1(fixed_batch_A)
        fixed_batch_BA_1 = self.generator_A_1(fixed_batch_B)
        fixed_sample_AB_1 = fixed_batch_AB_1[fixed_sample_index]
        fixed_sample_BA_1 = fixed_batch_BA_1[fixed_sample_index]

        AB_2 = self.generator_B_2(A)
        BA_2 = self.generator_A_2(B)

        ABA_2 = self.generator_A_2(AB_2)
        BAB_2 = self.generator_B_2(BA_2)

        fixed_batch_AB_2 = self.generator_B_2(fixed_batch_A)
        fixed_batch_BA_2 = self.generator_A_2(fixed_batch_B)
        fixed_sample_AB_2 = fixed_batch_AB_2[fixed_sample_index]
        fixed_sample_BA_2 = fixed_batch_BA_2[fixed_sample_index]
        fixed_sample_ABA_2 = self.generator_A_2(fixed_batch_AB_2)[fixed_sample_index]
        fixed_sample_BAB_2 = self.generator_B_2(fixed_batch_BA_2)[fixed_sample_index]

        # Reconstruction Loss
        recon_loss_A_1 = self.recon_criterion(ABA_1, A)
        recon_loss_B_1 = self.recon_criterion(BAB_1, B)

        recon_loss_A_2 = self.recon_criterion(ABA_2, A)
        recon_loss_B_2 = self.recon_criterion(BAB_2, B)

        fixed_sample_recon_loss_A_2 = self.recon_criterion(fixed_sample_ABA_2, fixed_batch_A[fixed_sample_index])
        fixed_sample_recon_loss_B_2 = self.recon_criterion(fixed_sample_BAB_2, fixed_batch_B[fixed_sample_index])

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

        ###########################################################33
        # One sample GAN losses
        A_dis_real_2, A_feats_real_2 = self.discriminator_A_2(fixed_batch_A[fixed_sample_index: fixed_sample_index + 1])
        A_dis_fake_2, A_feats_fake_2 = self.discriminator_A_2(fixed_batch_BA_2[fixed_sample_index: fixed_sample_index + 1])

        fixed_sample_dis_loss_A_2, fixed_sample_gen_loss_A_2 = self.get_gan_loss(A_dis_real_2, A_dis_fake_2)
        fixed_sample_fm_loss_A_2 = self.get_fm_loss(A_feats_real_2, A_feats_fake_2)

        # Real/Fake GAN Loss (B)
        B_dis_real_2, B_feats_real_2 = self.discriminator_B_2(fixed_batch_B[fixed_sample_index: fixed_sample_index + 1])
        B_dis_fake_2, B_feats_fake_2 = self.discriminator_B_2(fixed_batch_AB_2[fixed_sample_index: fixed_sample_index + 1])

        fixed_sample_dis_loss_B_2, fixed_sample_gen_loss_B_2 = self.get_gan_loss(B_dis_real_2, B_dis_fake_2)
        fixed_sample_fm_loss_B_2 = self.get_fm_loss(B_feats_real_2, B_feats_fake_2)

        # Correlation loss
        # Distance between generator 1 and generator 2's output
        correlation_loss_AB_2 = - self.correlation_criterion(fixed_sample_AB_2, self.to_no_grad_var(fixed_sample_AB_1))
        correlation_loss_BA_2 = - self.correlation_criterion(fixed_sample_BA_2, self.to_no_grad_var(fixed_sample_BA_1))

        # Total Loss
        if self.iters < self.args.gan_curriculum:
            rate = self.args.starting_rate
            correlation_rate = self.args.default_correlation_rate
        else:
            rate = self.args.default_rate
            correlation_rate = self.args.default_correlation_rate

        gen_loss_B_2 += self.args.indiv_gan_rate * fixed_sample_gen_loss_B_2
        fm_loss_B_2 += self.args.indiv_gan_rate * fixed_sample_fm_loss_B_2
        recon_loss_A_2 += self.args.indiv_gan_rate * fixed_sample_recon_loss_A_2

        gen_loss_A_2 += self.args.indiv_gan_rate * fixed_sample_gen_loss_A_2
        fm_loss_A_2 += self.args.indiv_gan_rate * fixed_sample_fm_loss_A_2
        recon_loss_B_2 += self.args.indiv_gan_rate * fixed_sample_recon_loss_B_2

        gen_loss_A_1_total = (gen_loss_B_1 * 0.1 + fm_loss_B_1 * 0.9) * (1. - rate) + recon_loss_A_1 * rate
        gen_loss_B_1_total = (gen_loss_A_1 * 0.1 + fm_loss_A_1 * 0.9) * (1. - rate) + recon_loss_B_1 * rate

        gen_loss_A_2_total = (gen_loss_B_2 * 0.1 + fm_loss_B_2 * 0.9) * (1. - rate) + recon_loss_A_2 * rate + correlation_loss_AB_2 * correlation_rate
        gen_loss_B_2_total = (gen_loss_A_2 * 0.1 + fm_loss_A_2 * 0.9) * (1. - rate) + recon_loss_B_2 * rate + correlation_loss_BA_2 * correlation_rate

        if self.args.model_arch == 'discogan':
            gen_loss_1 = gen_loss_A_1_total + gen_loss_B_1_total
            dis_loss_1 = dis_loss_A_1 + dis_loss_B_1
            gen_loss_2 = gen_loss_A_2_total + gen_loss_B_2_total
            dis_loss_2 = dis_loss_A_2 + dis_loss_B_2
            gen_loss_1 = gen_loss_1.detach()
            dis_loss_1 = dis_loss_1.detach()
            correlation_loss_AB_2 = correlation_loss_AB_2.detach()
            correlation_loss_BA_2 = correlation_loss_BA_2.detach()

        return gen_loss_1, dis_loss_1, gen_loss_2, dis_loss_2, \
                - correlation_loss_AB_2, - correlation_loss_BA_2,

    def run(self):

        self.initialize()
        self.iters = 0
        fixed_sample_batch = int(self.args.one_sample_index / self.args.batch_size)
        fixed_sample_index = self.args.one_sample_index % self.args.batch_size

        A_fixed_path = self.data_style_A[fixed_sample_batch * self.args.batch_size: (fixed_sample_batch + 1) * self.args.batch_size]
        B_fixed_path = self.data_style_B[fixed_sample_batch * self.args.batch_size: (fixed_sample_batch + 1) * self.args.batch_size]

        A_fixed, B_fixed = self.get_images(A_fixed_path, B_fixed_path)
        A_fixed = Variable(torch.FloatTensor(A_fixed))
        B_fixed = Variable(torch.FloatTensor(B_fixed))
        if self.cuda:
            A_fixed = A_fixed.cuda()
            B_fixed = B_fixed.cuda()

        self.generator_A_1 = torch.load(self.args.pretrained_generator_A_path)
        self.generator_B_1 = torch.load(self.args.pretrained_generator_B_path)
        self.discriminator_A_1 = torch.load(self.args.pretrained_discriminator_A_path)
        self.discriminator_B_1 = torch.load(self.args.pretrained_discriminator_B_path)

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

                _, _, self.gen_loss_2, self.dis_loss_2, correlation_loss_AB, correlation_loss_BA = self.calculate_losses(A, B, A_fixed, B_fixed, fixed_sample_index)

                current_correlation_loss_AB += correlation_loss_AB
                current_correlation_loss_BA += correlation_loss_BA

                self.finish_iteration()
                self.iters += 1

        return current_correlation_loss_AB/(self.n_batches), current_correlation_loss_BA/(self.n_batches)

    def finish_iteration(self):

        if self.iters % self.args.update_interval == 0:
            self.dis_loss_2.backward()
            self.optim_dis_2.step()
        else:
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
    model = GanBoundPerSample()
    model.run()
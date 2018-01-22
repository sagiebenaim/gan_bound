from torch.autograd import Variable
from dataset import *
from gan_bound_per_sample_discogan import GanBoundPerSample
from model import *

class GanBoundPerSample_distancegan(GanBoundPerSample):

    def distance(self, A, B):
        return torch.mean(torch.abs(A - B))

    def get_individual_distance_loss(self, A_i, A_j, AB_i, AB_j,
                                     B_i, B_j, BA_i, BA_j):

        distance_in_A = self.distance(A_i, A_j)
        distance_in_AB = self.distance(AB_i, AB_j)
        distance_in_B = self.distance(B_i, B_j)
        distance_in_BA = self.distance(BA_i, BA_j)

        if self.normalize_distances:
            distance_in_A = (distance_in_A - self.expectation_A) / self.std_A
            distance_in_AB = (distance_in_AB - self.expectation_B) / self.std_B
            distance_in_B = (distance_in_B - self.expectation_B) / self.std_B
            distance_in_BA = (distance_in_BA - self.expectation_A) / self.std_A

        return torch.abs(distance_in_A - distance_in_AB), torch.abs(distance_in_B - distance_in_BA)

    def get_self_distances(self, A, B, AB, BA):

        A_half_1, A_half_2 = torch.chunk(A, 2, dim=2)
        B_half_1, B_half_2 = torch.chunk(B, 2, dim=2)
        AB_half_1, AB_half_2 = torch.chunk(AB, 2, dim=2)
        BA_half_1, BA_half_2 = torch.chunk(BA, 2, dim=2)

        l_distance_A, l_distance_B = \
            self.get_individual_distance_loss(A_half_1, A_half_2,
                                              AB_half_1, AB_half_2,
                                              B_half_1, B_half_2,
                                              BA_half_1, BA_half_2)

        return l_distance_A, l_distance_B

    def get_distance_losses(self, A, B, AB, BA):

        As = torch.split(A, 1)
        Bs = torch.split(B, 1)
        ABs = torch.split(AB, 1)
        BAs = torch.split(BA, 1)

        loss_distance_A = 0.0
        loss_distance_B = 0.0
        num_pairs = 0
        min_length = min(len(As), len(Bs))

        for i in xrange(min_length - 1):
            for j in xrange(i + 1, min_length):
                num_pairs += 1
                loss_distance_A_ij, loss_distance_B_ij = \
                    self.get_individual_distance_loss(As[i], As[j],
                                                      ABs[i], ABs[j],
                                                      Bs[i], Bs[j],
                                                      BAs[i], BAs[j])

                loss_distance_A += loss_distance_A_ij
                loss_distance_B += loss_distance_B_ij

        loss_distance_A = loss_distance_A / num_pairs
        loss_distance_B = loss_distance_B / num_pairs

        return loss_distance_A, loss_distance_B

    def get_std(self, num_items, vars, expectation):

        num_pairs = 0
        std_sum = 0.0

        # If self distance computed std for top and bottom half
        if self.args.use_self_distance:
            for i in xrange(num_items):
                var_half_1, var_half_2 = torch.chunk(vars[i], 2, dim=2)
                std_sum += np.square(self.as_np(self.distance(var_half_1, var_half_2)) - expectation)
            return np.sqrt(std_sum / num_items)

        # Otherwise compute std for all pairs of images
        for i in xrange(num_items - 1):
            for j in xrange(i + 1, num_items):
                num_pairs += 1
                std_sum += np.square(self.as_np(self.distance(vars[i], vars[j])) - expectation)

        return np.sqrt(std_sum / num_pairs)

    def get_expectation(self, num_items, vars):

        num_pairs = 0
        distance_sum = 0.0

        # If self distance computed expectation for top and bottom half
        if self.args.use_self_distance:
            for i in xrange(num_items):
                # Split image to top and bottom half
                var_half_1, var_half_2 = torch.chunk(vars[i], 2, dim=2)
                distance_sum += self.as_np(self.distance(var_half_1, var_half_2))
            return distance_sum / num_items

        # Otherwise compute expectation for all pairs of images
        for i in xrange(num_items - 1):
            for j in xrange(i + 1, num_items):
                num_pairs += 1
                distance_sum += self.as_np(self.distance(vars[i], vars[j]))

        return distance_sum / num_pairs

    def set_expectation_and_std(self):

        max_items = self.args.max_items

        data_style_A, data_style_B = shuffle_data(self.data_style_A, self.data_style_B)

        if max_items < len(data_style_A):
            A_path = data_style_A[0:max_items]
        else:
            A_path = data_style_A

        if max_items < len(data_style_B):
            B_path = data_style_B[0:max_items]
        else:
            B_path = data_style_B

        dataset_A, dataset_B = self.get_images(A_path, B_path)

        A_vars = []
        num_vars_A = 0

        for step_data_a, data in enumerate(dataset_A):

            if step_data_a >= max_items:
                break

            A = Variable(torch.FloatTensor(data), volatile=True)
            if self.cuda:
                A = A.cuda()

            A_vars.append(A)
            num_vars_A += 1

        B_vars = []
        num_vars_B = 0
        for step_data_b, data in enumerate(dataset_B):

            if step_data_b >= max_items:
                break

            B = Variable(torch.FloatTensor(data), volatile=True)
            if self.cuda:
                B = B.cuda()

            B_vars.append(B)
            num_vars_B += 1

        self.expectation_A = self.get_expectation(num_vars_A, A_vars)[0].astype(float)
        self.expectation_B = self.get_expectation(num_vars_B, B_vars)[0].astype(float)
        self.std_A = self.get_std(num_vars_A, A_vars, self.expectation_A)[0].astype(float)
        self.std_B = self.get_std(num_vars_B, B_vars, self.expectation_B)[0].astype(float)

        print('Expectation for dataset A: %f' % self.expectation_A)
        print('Expectation for dataset B: %f' % self.expectation_B)
        print('Std for dataset A: %f' % self.std_A)
        print('Std for dataset B: %f' % self.std_B)

    def calculate_losses(self, A, B, fixed_batch_A, fixed_batch_B, fixed_sample_index):

        AB_1 = self.generator_B_1(A)
        BA_1 = self.generator_A_1(B)

        fixed_batch_AB_1 = self.generator_B_1(fixed_batch_A)
        fixed_batch_BA_1 = self.generator_A_1(fixed_batch_B)
        fixed_sample_AB_1 = fixed_batch_AB_1[fixed_sample_index]
        fixed_sample_BA_1 = fixed_batch_BA_1[fixed_sample_index]

        AB_2 = self.generator_B_2(A)
        BA_2 = self.generator_A_2(B)

        fixed_batch_AB_2 = self.generator_B_2(fixed_batch_A)
        fixed_batch_BA_2 = self.generator_A_2(fixed_batch_B)
        fixed_sample_AB_2 = fixed_batch_AB_2[fixed_sample_index]
        fixed_sample_BA_2 = fixed_batch_BA_2[fixed_sample_index]
        loss_distance_A_1, loss_distance_B_1 = self.get_self_distances(A, B, AB_1, BA_1)
        loss_distance_A_2, loss_distance_B_2 = self.get_self_distances(A, B, AB_2, BA_2)

        fixed_sample_A_list = fixed_batch_A[fixed_sample_index: fixed_sample_index + 1]
        fixed_sample_B_list = fixed_batch_B[fixed_sample_index: fixed_sample_index + 1]
        fixed_sample_BA_2_list = fixed_batch_BA_2[fixed_sample_index: fixed_sample_index + 1]
        fixed_sample_AB_2_list = fixed_batch_AB_2[fixed_sample_index: fixed_sample_index + 1]
        fixed_sample_loss_distance_A_2, fixed_sample_loss_distance_B_2 = self.get_self_distances(fixed_sample_A_list, fixed_sample_B_list, fixed_sample_AB_2_list, fixed_sample_BA_2_list)

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
        loss_distance_B_2 += self.args.indiv_gan_rate * fixed_sample_loss_distance_B_2

        gen_loss_A_2 += self.args.indiv_gan_rate * fixed_sample_gen_loss_A_2
        fm_loss_A_2 += self.args.indiv_gan_rate * fixed_sample_fm_loss_A_2
        loss_distance_A_2 += self.args.indiv_gan_rate * fixed_sample_loss_distance_A_2

        gen_loss_A_1_total = (gen_loss_B_1 * 0.1 + fm_loss_B_1 * 0.9) * (1. - rate) + loss_distance_A_1 * rate
        gen_loss_B_1_total = (gen_loss_A_1 * 0.1 + fm_loss_A_1 * 0.9) * (1. - rate) + loss_distance_B_1 * rate

        gen_loss_A_2_total = (gen_loss_B_2 * 0.1 + fm_loss_B_2 * 0.9) * (1. - rate) + loss_distance_A_2 * rate + correlation_loss_AB_2 * correlation_rate
        gen_loss_B_2_total = (gen_loss_A_2 * 0.1 + fm_loss_A_2 * 0.9) * (1. - rate) + loss_distance_A_2 * rate + correlation_loss_BA_2 * correlation_rate

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
        self.normalize_distances = not self.args.unnormalized_distances

        if self.normalize_distances:
            self.set_expectation_and_std()

        return super(GanBoundPerSample_distancegan, self).run()


if __name__ == "__main__":
    model = GanBoundPerSample_distancegan()
    model.run()
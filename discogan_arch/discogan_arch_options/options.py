import argparse

class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch implementation of DistanceGAN based on DiscoGAN')
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
        self.parser.add_argument('--task_name', type=str, default='facescrub', help='Set data name')
        self.parser.add_argument('--epoch_size', type=int, default=500, help='Set epoch size')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
        self.parser.add_argument('--model_arch', type=str, default='discogan',
                            help='choose among gan/recongan/discogan/distancegan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN. distancegan')
        self.parser.add_argument('--image_size', type=int, default=64,
                            help='Image size. 64 for every experiment in the paper')
        self.parser.add_argument('--gan_curriculum', type=int, default=10000,
                            help='Strong GAN loss for certain period at the beginning')
        self.parser.add_argument('--starting_rate', type=float, default=0.01,
                            help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
        self.parser.add_argument('--default_rate', type=float, default=0.5,
                            help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')
        self.parser.add_argument('--style_A', type=str, default=None,
                            help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
        self.parser.add_argument('--style_B', type=str, default=None,
                            help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
        self.parser.add_argument('--constraint', type=str, default=None,
                            help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
        self.parser.add_argument('--constraint_type', type=str, default=None,
                            help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
        self.parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')
        self.parser.add_argument('--update_interval', type=int, default=3, help='')
        self.parser.add_argument('--log_interval', type=int, default=50,
                            help='Print loss values every log_interval iterations.')
        self.parser.add_argument('--image_save_interval', type=int, default=1000,
                            help='Save test results every log_interval iterations.')
        self.parser.add_argument('--model_save_interval', type=int, default=10000,
                            help='Save models every log_interval iterations.')
        self.parser.add_argument('--result_path', type=str, default='./results/')
        self.parser.add_argument('--model_path', type=str, default='./models/')
        self.parser.add_argument('--use_self_distance', action='store_true', help="Use distance for top and bottom half of the image")
        self.parser.add_argument('--unnormalized_distances', action='store_true', help='Do not normalize distances by expecatation and std')
        self.parser.add_argument('--max_items', type=int, default=300, help='Maximum number of items to use for expectation and std calculation')
        self.parser.add_argument('--use_reconst_loss', action='store_true',
                                 help='Add reconstruction loss in addition to distance loss')
        self.parser.add_argument('--num_layers', type=int, default=4, help='Number of convolutional layers in G (equal number of deconvolutional layers exist)')
        self.parser.add_argument('--starting_correlation_rate', type=float, default=1, help='Starting Correlation rate between the G_1 and G_2')
        self.parser.add_argument('--default_correlation_rate', type=float, default=1, help='Default Correlation rate between the G_1 and G_2')
        self.parser.add_argument('--number_of_samples', type=int, default=500, help='Restrict the number of samples to use')
        self.parser.add_argument('--not_all_samples', action='store_true', help='Do not use all samples, only *num_of_samples* amount')
        self.parser.add_argument('--one_sample_index', type=int, default=0, help='Index of the sample for which we want to find the error')
        self.parser.add_argument('--indiv_gan_rate', type=float, default=1, help='Individial Sample GAN loss weight for per-sample error')
        self.parser.add_argument('--pretrained_generator_A_path', type=str, default='./models/model_gen_A-3', help="Pretrained G_1 generator A to B, for per sample error experiments")
        self.parser.add_argument('--pretrained_generator_B_path', type=str, default='./models/model_gen_B-3', help="Pretrained G_1 B to A, for per sample error experiments")
        self.parser.add_argument('--pretrained_discriminator_A_path', type=str, default='./models/model_dis_A-3', help="Pretrained D_1 discriminator A to B, for per sample error experiments")
        self.parser.add_argument('--pretrained_discriminator_B_path', type=str, default='./models/model_dis_B-3', help="Pretrained D_1 discriminator B to a, for per sample error experiments")
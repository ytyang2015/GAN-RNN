import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.autograd import Variable

#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # Batch size.
    N = logits_real.size()

    # Target label vector, the discriminator should be aiming
    true_labels = Variable(torch.ones(N)).type(dtype)

    # Discriminator loss has 2 parts: how well it classifies real images and how well it
    # classifies fake images.
    real_image_loss = bce_loss(logits_real, true_labels)
    fake_image_loss = bce_loss(logits_fake, 1 - true_labels)

    loss = real_image_loss + fake_image_loss


    ##########       END      ##########

    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # Batch size.
    N = logits_fake.size()

    # Generator is trying to make the discriminator output 1 for all its images.
    # So we create a 'target' label vector of ones for computing generator loss.
    true_labels = Variable(torch.ones(N)).type(dtype)

    # Compute the generator loss compraing
    loss = bce_loss(logits_fake, true_labels)


    ##########       END      ##########

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################
    N = scores_real.size()

    true_labels = Variable(torch.ones(N)).type(dtype)

    fake_image_loss = (torch.mean((scores_real - true_labels)**2))
    real_image_loss = (torch.mean((scores_fake)**2))

    loss = 0.5*fake_image_loss + 0.5*real_image_loss


    ##########       END      ##########

    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################
    N = scores_fake.size()

    true_labels = Variable(torch.ones(N)).type(dtype)

    loss = 0.5 * ((torch.mean((scores_fake - true_labels)**2)))

    ##########       END      ##########

    return loss

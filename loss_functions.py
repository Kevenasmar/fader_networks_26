import torch
import torch.nn.functional as F

# Auto-Encoder Loss 
def reconstruction_loss(x, x_rec):
    squared_error = (x_rec - x) ** 2
    mse_loss = torch.mean(squared_error)
    return mse_loss

# Discriminator Loss
def discriminator_loss(encoder, discriminator, x, y):
    """
    Discriminator objective: maximize log P(y | E(x))
    => minimize negative log-likelihood.

    encoder: E
    discriminator: D, outputs logits for each attribute
    x: (B, C, H, W)
    y: (B, K) with 0/1 values (float or int)
    """
    with torch.no_grad():
        z = encoder(x)  # (B, latent_dim)

    logits = discriminator(z)  # (B, K) logits
    loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="mean")
    return loss

# Adversial loss
# def adversarial_loss(
#     encoder,
#     decoder,
#     discriminator,
#     x,
#     y,                
#     lambda_E,
#     reconstruction_loss_fn
# ):
#     """
#     Notes:
#     - We update encoder+decoder here, so gradients should flow through E and Dec.
#     - Discriminator parameters are *frozen* during this step (but we still use its forward).
#     """

#     z = encoder(x)  
#     x_rec = decoder(z, y)

#     recon = reconstruction_loss(x, x_rec)

#     # Adversarial term: make D predict the WRONG attribute (1 - y)
#     for p in discriminator.parameters():
#         p.requires_grad_(False)

#     logits = discriminator(z)          
#     y_flipped = 1.0 - y.float()        

#     adv = F.binary_cross_entropy_with_logits(
#         logits, y_flipped, reduction="mean"
#     )

#     for p in discriminator.parameters():
#         p.requires_grad_(True)

#     loss = recon + lambda_E * adv

#     return loss, {"recon": recon.detach(), "adv": adv.detach()}


# import torch.nn.functional as F

def adversarial_loss(encoder, decoder, discriminator, x, y_onehot, y_bin, lambda_E, reconstruction_loss_fn):
    z = encoder(x)
    x_rec = decoder(z, y_onehot)
    
    recon = reconstruction_loss_fn(x, x_rec)
    
    # discriminator is assumed frozen in the training loop (not here)
    logits = discriminator(z)              # (B,1)
    y_flip = 1.0 - y_bin.float()          # (B,1)
    
    adv = F.binary_cross_entropy_with_logits(logits, y_flip, reduction="mean")
    loss = recon + lambda_E * adv
    return loss, {"recon": recon.detach(), "adv": adv.detach()}


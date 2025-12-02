import torch
import torch.nn.functional as F

from models import Generator, Discriminator
from cycle_components import HQCycle

# Small end-to-end smoke: generator + discriminator + HQCycle cycle loss + optimizers

def label2onehot(labels, dim):
    # labels: (batch, V, V) integer
    bsize = labels.shape[0]
    out = torch.zeros(*labels.shape, dim)
    for i in range(bsize):
        out[i].scatter_(-1, labels[i].unsqueeze(-1), 1.)
    return out


def main():
    device = torch.device('cpu')
    batch = 2
    vertexes = 5
    atom_types = 4
    bond_types = 4
    z_dim = 4

    # instantiate models
    G = Generator(conv_dims=[16], z_dim=z_dim, vertexes=vertexes, edges=bond_types, nodes=atom_types, dropout=0.)
    D = Discriminator(conv_dim=([8, 8], 16, [32]), m_dim=atom_types, b_dim=bond_types, dropout=0.)

    # HQCycle uses intermediate_dim=16 and QDI with batched execution
    hc = HQCycle(input_dim=vertexes*atom_types + vertexes*vertexes*bond_types,
                 intermediate_dim=16, z_dim=z_dim, classical_hidden=[32], qdi_kwargs={'n_reps':1, 'n_layers':1, 'n_qubits':z_dim, 'batch':True})

    G.train(); D.train(); hc.train()

    G.to(device); D.to(device); hc.to(device)

    d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    g_opt = torch.optim.Adam(list(G.parameters()) + list(hc.parameters()), lr=1e-3)

    # create fake real data
    a = torch.randint(0, bond_types, (batch, vertexes, vertexes))
    x = torch.randint(0, atom_types, (batch, vertexes))

    a_tensor = label2onehot(a, bond_types).to(device)
    x_tensor = label2onehot(x, atom_types).to(device)

    z = torch.randn(batch, z_dim).to(device)

    # Discriminator real
    logits_real, features_real = D(a_tensor, None, x_tensor)
    d_loss_real = - torch.mean(logits_real)

    # Discriminator fake
    edges_logits, nodes_logits = G(z)
    edges_hat = F.softmax(edges_logits, dim=-1)
    nodes_hat = F.softmax(nodes_logits, dim=-1)

    logits_fake, features_fake = D(edges_hat, None, nodes_hat)
    d_loss_fake = torch.mean(logits_fake)

    d_loss = d_loss_fake + d_loss_real
    d_opt.zero_grad(); d_loss.backward(); d_opt.step()

    # Generator step
    edges_logits, nodes_logits = G(z)
    edges_hat = F.softmax(edges_logits, dim=-1)
    nodes_hat = F.softmax(nodes_logits, dim=-1)

    logits_fake, _ = D(edges_hat, None, nodes_hat)
    g_loss_fake = - torch.mean(logits_fake)

    # cycle loss
    nodes_flat = nodes_hat.view(batch, -1)
    edges_flat = edges_hat.view(batch, -1)
    cycle_input = torch.cat([nodes_flat, edges_flat], dim=1)
    z_hat = hc(cycle_input)
    cycle_loss = F.mse_loss(z, z_hat)

    g_loss = g_loss_fake + 0.1 * cycle_loss
    g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    print('d_loss:', d_loss.item(), 'g_loss:', g_loss.item(), 'cycle_loss:', cycle_loss.item())

if __name__ == '__main__':
    main()

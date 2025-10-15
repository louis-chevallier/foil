# from chatgpt :)

Oui 👍 — voici un exemple **complet et pédagogique** de **simulateur 2D de fluide eulérien** écrit en **Python** avec **PyTorch**.

Ce code illustre le principe du **solver de Navier–Stokes incompressible** sur grille régulière, avec advection semi-lagrangienne, diffusion, projection de pression (Jacobi), et visualisation avec `matplotlib`.

---

## 💨 Principe général

L’équation de Navier–Stokes discrétisée (en formulation eulérienne) :
[
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f}
]
[
\nabla\cdot\mathbf{u} = 0
]

où :

* ( \mathbf{u} ) = champ de vitesse (u, v)
* ( p ) = pression
* ( \nu ) = viscosité
* ( \mathbf{f} ) = forces externes (ex. gravité, injection de vitesse)

---

## 🧠 Objectif

Ce programme :

* utilise **PyTorch** pour les calculs sur GPU (si dispo),
* simule un fluide 2D sur une grille fixe,
* visualise la **densité de fluide** en temps réel,
* implémente : advection, diffusion, projection de pression.

---

## 🧩 Code complet

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# =======================
# Paramètres du simulateur
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
N = 128           # résolution de la grille
dt = 0.1          # pas de temps
viscosity = 0.001 # coefficient de diffusion
iterations = 20   # itérations pour la projection de pression

# Champs du fluide
u = torch.zeros((1, 1, N, N), device=device)  # vitesse X
v = torch.zeros((1, 1, N, N), device=device)  # vitesse Y
density = torch.zeros((1, 1, N, N), device=device)

# ===============
# Fonctions utiles
# ===============
def advect(field, u, v):
    """Advection semi-lagrangienne : backtrace les particules."""
    n, c, h, w = field.shape
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    x = x.float() - u[0,0]*dt
    y = y.float() - v[0,0]*dt

    # Clamp des coordonnées pour rester dans le domaine
    x = torch.clamp(x, 0, w-1)
    y = torch.clamp(y, 0, h-1)

    # Normalisation en coordonnées de grille [−1,1]
    x = (x / (w - 1)) * 2 - 1
    y = (y / (h - 1)) * 2 - 1
    grid = torch.stack((x, y), dim=-1)

    return F.grid_sample(field, grid.unsqueeze(0), align_corners=True)

def diffuse(field, diff):
    """Diffusion simple par itérations de Jacobi."""
    a = dt * diff * N * N
    result = field.clone()
    for _ in range(iterations):
        result = (field + a * (
            F.pad(result, (1,1,1,1), mode='replicate')[:,:,1:-1,2:] +
            F.pad(result, (1,1,1,1), mode='replicate')[:,:,1:-1,:-2] +
            F.pad(result, (1,1,1,1), mode='replicate')[:,:,2:,1:-1] +
            F.pad(result, (1,1,1,1), mode='replicate')[:,:,:-2,1:-1]
        )) / (1 + 4 * a)
    return result

def project(u, v):
    """Rend le champ de vitesse incompressible (projection de pression)."""
    h = 1.0 / N
    div = -0.5 * h * (
        u[:,:, :,2:] - u[:,:,:,:-2] +
        v[:,:,2:,:] - v[:,:,:-2,:]
    )
    div = F.pad(div, (1,1,1,1), mode='constant', value=0)
    p = torch.zeros_like(u)

    for _ in range(iterations):
        p = (div + (
            F.pad(p, (1,1,1,1), mode='replicate')[:,:,1:-1,2:] +
            F.pad(p, (1,1,1,1), mode='replicate')[:,:,1:-1,:-2] +
            F.pad(p, (1,1,1,1), mode='replicate')[:,:,2:,1:-1] +
            F.pad(p, (1,1,1,1), mode='replicate')[:,:,:-2,1:-1]
        )) / 4

    u = u - 0.5 * (p[:,:,:,2:] - p[:,:,:,:-2]) / h
    v = v - 0.5 * (p[:,:,2:,:] - p[:,:,:-2,:]) / h

    # Réajuster la taille après découpe des bords
    u = F.pad(u, (1,1,1,1))[:,:,1:-1,1:-1]
    v = F.pad(v, (1,1,1,1))[:,:,1:-1,1:-1]
    return u, v

# =================
# Boucle de simulation
# =================
plt.ion()
fig, ax = plt.subplots()

for frame in range(500):
    # Injection de densité et vitesse au centre
    cx, cy = N//2, N//2
    density[:,:,cy-2:cy+2, cx-2:cx+2] += 5.0
    u[:,:,cy-2:cy+2, cx-2:cx+2] += torch.randn(1,1,4,4, device=device)*0.2
    v[:,:,cy-2:cy+2, cx-2:cx+2] += torch.randn(1,1,4,4, device=device)*0.2

    # Étapes principales du solveur
    u = diffuse(u, viscosity)
    v = diffuse(v, viscosity)

    u, v = project(u, v)

    u_adv = advect(u, u, v)
    v_adv = advect(v, u, v)
    density = advect(density, u, v)

    u, v = project(u_adv, v_adv)

    # Affichage
    ax.clear()
    ax.imshow(density[0,0].cpu().numpy(), cmap='plasma', origin='lower')
    ax.set_title(f"Simulation fluide - frame {frame}")
    ax.axis('off')
    plt.pause(0.01)

plt.ioff()
plt.show()
```

---

## 🧩 Détails techniques

| Étape             | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| **Advection**     | Transporte les valeurs selon la vitesse via `grid_sample`. |
| **Diffusion**     | Lisse le champ selon la viscosité (méthode de Jacobi).     |
| **Projection**    | Calcule la pression pour forcer la divergence nulle.       |
| **Visualisation** | Affiche la densité (comme de la fumée).                    |

---

## ⚡ Pour aller plus loin

Tu peux :

* Ajouter un **champ de force** externe (ex. gravité)
* Utiliser un **colormap vectoriel** pour visualiser les vitesses
* Adapter le code pour **3D** (plus lent mais faisable avec PyTorch)
* Convertir le rendu en **animation interactive** avec `pygame`

---

Souhaites-tu que je t’ajoute une **version interactive avec la souris** (pour injecter du fluide en temps réel) ?

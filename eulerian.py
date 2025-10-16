
import sys, os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utillc import *

from tkinter import Tk, Label
from PIL import Image, ImageTk

print_everything()

# =======================
# Paramètres du simulateur
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
EKOX(device)
N = 128*2			  # résolution de la grille
dt = 0.1		  # pas de temps
viscosity = 0.001 # coefficient de diffusion
iterations = 20	  # itérations pour la projection de pression

todev = lambda x : x.cuda() if device == 'cuda' else x
T= torch.float32

# Champs du fluide
u = torch.zeros((1, 1, N, N), device=device)  # vitesse X
v = torch.zeros((1, 1, N, N), device=device)  # vitesse Y
density = torch.zeros((1, 1, N, N), device=device)

# v : vertical, u : horizontal
# u[batch,depth, index vertical , index horizontal] += 1./10

obstacle = torch.zeros((1, 1, N, N), device=device).byte()
obstacle[:,:,N//3 : N//3 + N//10, N//5 : N//5 + N//10] = 1
		 
EKOX(u.dtype)
#sys.exit(0)

fu = todev(torch.nn.Conv2d(1, 1, (2,1), padding='same'))
fu.weight.data=todev(torch.tensor([[[[-1.], [1.]]]]))
fu.bias.data = todev(torch.zeros((1)))

right = todev(torch.nn.Conv2d(1, 1, (2,1), padding='same'))
right.weight.data=torch.tensor([[[[0.], [1.]]]])
right.bias.data = torch.zeros((1))

fv = todev(torch.nn.Conv2d(1, 1, (1,2), padding='same'))
fv.weight.data = todev(torch.tensor([[[[1., -1.]]]]))
fv.bias.data = todev(torch.zeros((1)))

up = todev(torch.nn.Conv2d(1, 1, (1,2), padding='same'))
up.weight.data = torch.tensor([[[[1., 0.]]]])
up.bias.data = torch.zeros((1))

# ===============
# Fonctions utiles
# ===============
def advect(field, u, v):
	"""Advection semi-lagrangienne : backtrace les particules."""
	n, c, h, w = field.shape
	#EKOX(field.shape)
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
	#EKOX(field.shape)
	for _ in range(iterations):
		result = (field + a * (
			F.pad(result, (1,1,1,1), mode='replicate')[:,:,1:-1,2:] +
			F.pad(result, (1,1,1,1), mode='replicate')[:,:,1:-1,:-2] +
			F.pad(result, (1,1,1,1), mode='replicate')[:,:,2:,1:-1] +
			F.pad(result, (1,1,1,1), mode='replicate')[:,:,:-2,1:-1]
		)) / (1 + 4 * a)
	#EKOX(result.shape)
	return result

def project(u, v):
	"""Rend le champ de vitesse incompressible (projection de pression)."""
	h = 1.0 / N

	#EKOX(u[:,:, :,2:].shape)
	#EKOX(u[:,:, :,:-2].shape)
	#EKOX(v[:,:,2:,:].shape)
	#EKOX(v[:,:,:-2,:].shape)

	div	 = -0.5 * h * (fu(u) + fv(v))
	#EKOX(div.shape)
	"""
	div = -0.5 * h * (
		u[:,:, :,2:] - u[:,:,:,:-2] +
		v[:,:,2:,:] - v[:,:,:-2,:]
	)
	"""
	#div = F.pad(div, (1,1,1,1), mode='constant', value=0)
	#EKOX(div.shape)	
	p = torch.zeros_like(u)

	for _ in range(iterations):
		p = (div + (
			F.pad(p, (1,1,1,1), mode='replicate')[:,:,1:-1,2:] +
			F.pad(p, (1,1,1,1), mode='replicate')[:,:,1:-1,:-2] +
			F.pad(p, (1,1,1,1), mode='replicate')[:,:,2:,1:-1] +
			F.pad(p, (1,1,1,1), mode='replicate')[:,:,:-2,1:-1]
		)) / 4
	#EKOX(p.shape)

	"""
	u = u - 0.5 * (p[:,:,:,2:] - p[:,:,:,:-2]) / h
	v = v - 0.5 * (p[:,:,2:,:] - p[:,:,:-2,:]) / h
	"""
	u = u - 0.5 * fu(p)/h
	v = v - 0.5 * fv(p)/h
	#EKOX(u.shape)
	#EKOX(v.shape)
	
	# Réajuster la taille après découpe des bords
	u = F.pad(u, (1,1,1,1))[:,:,1:-1,1:-1]
	v = F.pad(v, (1,1,1,1))[:,:,1:-1,1:-1]

	#EKOX(u.shape)
	#EKOX(v.shape)

	
	return u, v

# =================
# Boucle de simulation
# =================


root = Tk()
root.title("Simulation Fluide 2D (PyTorch + Tkinter)")
label = Label(root)
label.pack()
txt = Label(root, 
			text ="count",
			bg="lightblue"				 )

# Pack the label into the window
txt.pack(pady=20)  
count = 0
M=2

def update_frame():
	global count, u, v, density
	count += 1
	txt['text'] = "count %d" % count

	# flux entrant
	if False :
		# Injection de densité et vitesse au centre
		cx, cy = N//2, N//2
		density[:,:,cy-2:cy+2, cx-2:cx+2] += 5.0
		u[:,:,cy-2:cy+2, cx-2:cx+2] += torch.randn(1,1,4,4, device=device)*0.2
		v[:,:,cy-2:cy+2, cx-2:cx+2] += torch.randn(1,1,4,4, device=device)*0.2

	density[:,:,:, 0] = M
	u[:,:, :, 0] += 1./10


	# obstacle
	solid = obstacle.bool()
	u[solid] = 0.0
	v[solid] = 0.0	

	
	# Étapes principales du solveur
	u = diffuse(u, viscosity)
	v = diffuse(v, viscosity)

	u, v = project(u, v)

	u_adv = advect(u, u, v)
	v_adv = advect(v, u, v)
	density = advect(density, u, v)

	u, v = project(u_adv, v_adv)

	if count == -200 :
		plt.imshow(density[0, 0].detach().cpu().numpy()); plt.show()


	img = density[0, 0].clamp(0, M) / M
	img = (img * 255).byte().cpu().numpy()

	oo = obstacle.bool().cpu().numpy()[0,0]
	img[oo] = 122
	
	img = Image.fromarray(img, mode="L").resize((256, 256))
	img_tk = ImageTk.PhotoImage(img)

	# Mise à jour Tkinter
	label.imgtk = img_tk
	label.configure(image=img_tk)

	root.after(20, update_frame)

def key_press(event):
	key = event.char
	if key == 'q' : sys.exit()
	
root.bind('<Key>', key_press)	
# Lancement de la boucle
update_frame()

with torch.no_grad():
		root.mainloop()


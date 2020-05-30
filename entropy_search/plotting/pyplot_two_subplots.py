# Copyright 2020 Max Planck Society. All rights reserved.
# 
# Author: Alonso Marco Valle (amarcovalle/alonrot) amarco(at)tuebingen.mpg.de
# Affiliation: Max Planck Institute for Intelligent Systems, Autonomous Motion
# Department / Intelligent Control Systems
# 
# This file is part of EntropySearchCpp.
# 
# EntropySearchCpp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
# 
# EntropySearchCpp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# EntropySearchCpp.  If not, see <http://www.gnu.org/licenses/>.
#
#
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import yaml
import pdb

def plot_ES( docs ):

	z_plot 	= np.array(docs["z_plot"])
	f_true 	= np.array(docs["f_true"])
	mpost 	= np.array(docs["mpost"])
	stdpost = np.array(docs["stdpost"])
	Xdata 	= np.array(docs["Xdata"])
	Ydata 	= np.array(docs["Ydata"])
	dH_plot = np.array(docs["dH_plot"])
	EdH_max = np.array(docs["EdH_max"])
	x_next 	= np.array(docs["x_next"])
	mu_next = np.array(docs["mu_next"])

	assert z_plot.shape[1] == 1, "This plotting tool is tailored for 1D"

	fig = plt.figure(num=1, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
	plt.subplot(211)
	plt.grid(True)

	Npoints = z_plot[:,0].shape[0]
	z_plot = z_plot[:,0]
	if mpost.shape[0] > Npoints:
		mpost = mpost[0:Npoints]
		stdpost = stdpost[0:Npoints]
		f_true = f_true[0:Npoints]
		dH_plot = dH_plot[0:Npoints]

	# GP posterior:
	plt.plot(z_plot, mpost, 'r', lw=1)
	plt.fill_between(z_plot, mpost-2*stdpost, mpost+2*stdpost, alpha=0.2, color='r')

	# Prior function:
	plt.plot(z_plot,f_true,'k--')

	# Data:
	plt.plot(Xdata,Ydata,'bo')

	# Next evaluations:
	plt.plot(x_next,mu_next,'go')

	plt.title("Entropy Search", fontsize=12)
	# plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

	plt.subplot(212)
	plt.grid(True)
	plt.plot(z_plot, dH_plot, 'b')
	plt.plot(x_next, EdH_max, 'bo')

	plt.show(block=False)
	plt.pause(0.2)

# Main:
current_numiter = 0
numiter = 0
while current_numiter < 10 :

	stream 	= open("examples/runES_onedim/output/tmp.yaml", "r")
	docs 		= yaml.load(stream)
	numiter = np.array(docs["numiter"])

	# Update:
	if current_numiter != numiter :
		plot_ES(docs)
		current_numiter = numiter



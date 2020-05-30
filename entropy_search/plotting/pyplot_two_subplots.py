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
import threading
import logging
import time
logger = logging.getLogger(__name__)


class ReadAndPlot():

	def __init__(self,path2data):
		self.path2data = path2data
		self.my_node = None

		hdl_fig = plt.figure(figsize=(12, 8))
		hdl_fig.suptitle("Bayesian optimization")
		grid_size = (2,1)
		self.axes_GPobj = plt.subplot2grid(grid_size, (0,0), colspan=1,fig=hdl_fig)
		self.axes_acqui = plt.subplot2grid(grid_size, (1,0), colspan=1,fig=hdl_fig)

		logger.info("Loading plotting data from {0:s} ...".format(self.path2data))

	def plot_ES(self):

		# Convert to numpy array:
		z_plot 	= np.array(self.my_node["z_plot"])
		f_true 	= np.array(self.my_node["f_true"])
		mpost 	= np.array(self.my_node["mpost"])
		stdpost = np.array(self.my_node["stdpost"])
		Xdata 	= np.array(self.my_node["Xdata"])
		Ydata 	= np.array(self.my_node["Ydata"])
		dH_plot = np.array(self.my_node["dH_plot"])
		EdH_max = np.array(self.my_node["EdH_max"])
		x_next 	= np.array(self.my_node["x_next"])
		mu_next = np.array(self.my_node["mu_next"])

		assert z_plot.shape[1] == 1, "This plotting tool is tailored for 1D"

		# Adjust data shape:
		Npoints = z_plot[:,0].shape[0]
		z_plot = z_plot[:,0]
		if mpost.shape[0] > Npoints:
			mpost = mpost[0:Npoints]
			stdpost = stdpost[0:Npoints]
			f_true = f_true[0:Npoints]
			dH_plot = dH_plot[0:Npoints]

		# GP posterior:
		self.axes_GPobj.cla()
		self.axes_GPobj.grid(True)
		self.axes_GPobj.set_xlim([0,1])
		self.axes_GPobj.plot(z_plot, mpost, 'r', lw=1)
		self.axes_GPobj.fill_between(z_plot, mpost-2*stdpost, mpost+2*stdpost, alpha=0.2, color='r')

		# Prior function:
		self.axes_GPobj.plot(z_plot,f_true,'k--')

		# Data:
		self.axes_GPobj.plot(Xdata,Ydata,'bo')

		# Next evaluations:
		self.axes_GPobj.plot(x_next,mu_next,'go')

		self.axes_GPobj.set_title("Gaussian process", fontsize=12)
		# plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

		self.axes_acqui.cla()
		self.axes_acqui.grid(True)
		self.axes_acqui.set_xlim([0,1])
		self.axes_acqui.plot(z_plot, dH_plot, 'b')
		self.axes_acqui.plot(x_next, EdH_max, 'bo')
		self.axes_acqui.set_title("Acquisition function", fontsize=12)

		plt.show(block=False)
		plt.pause(0.5)

	def read_from_yaml(self):

		while True:

			try:
				stream 	= open(self.path2data, "r")
				self.my_node = yaml.load(stream,Loader=yaml.UnsafeLoader)
			except Exception as inst:
				logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
				logger.info("Data corrupted or non-existent!!!")
				# pdb.set_trace()

			time.sleep(0.5)

	def run(self):

		thread_reading_plotdata = threading.Thread(target=self.read_from_yaml)
		thread_reading_plotdata.setDaemon(True)
		thread_reading_plotdata.start()
		time.sleep(0.5)

		while True:
			self.plot_ES()

def main():

	rap = ReadAndPlot(path2data="examples/runES_onedim/output/tmp.yaml")
	rap.run()


if __name__ == "__main__":

	main()



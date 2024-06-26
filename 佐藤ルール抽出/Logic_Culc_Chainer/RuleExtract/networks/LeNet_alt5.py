import sys
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.datasets import tuple_dataset
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import cupy as cp
from collections import OrderedDict
import wx

class LeNet_alt5(Chain):
	def __init__(self):
		super(LeNet_alt5, self).__init__()
		self.output = {}
		with self.init_scope():
			self.conv1 = L.Convolution2D(1, 17, 9)
			self.conv2 = L.Convolution2D(17, 35, 9)
			self.affine1 = L.Linear(35, 35)
			self.affine2 = L.Linear(35, 10)

	def __call__(self, x):
		out = x
		self.output = {}

		out = self.conv1(out)
		self.output["Conv1"] = out.array
		out =  F.max_pooling_2d(out, 2)
		self.output["Pool1"] = out.array
		out = self.conv2(out)
		self.output["Conv2"] = out.array
		out = F.sigmoid(out)
		self.output["Sigmoid1"] = out.array
		out =  F.max_pooling_2d(out, 2)
		self.output["Pool2"] = out.array
		out = self.affine1(out)
		self.output["Affine1"] = out.array
		out = F.sigmoid(out)
		self.output["Sigmoid2"] = out.array
		out = self.affine2(out)
		self.output["Affine2"] = out.array
		return out

	def get_output(self):
		return self.output

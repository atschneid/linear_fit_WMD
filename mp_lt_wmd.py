#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue April 17 2018

@author: a.schneider
"""

import numpy as np
from random import sample
import itertools
import sys
import time
import pickle
import scipy.io as sio
from pyemd import emd, emd_with_flow

import multiprocessing as mp

def global_mp_wmd( arg, **kwarg ):
    return lin_transform_wmd.mp_wmd( *arg, **kwarg )

def global_mp_grads_plus( arg, **kwarg ):
    return lin_transform_wmd.set_one_grad_plus( *arg, **kwarg )

def global_mp_grads_minus( arg, **kwarg ):
    return lin_transform_wmd.set_one_grad_minus( *arg, **kwarg )

def sample_data( fname, scale, random=False ):
        data = sio.loadmat(fname)
        [BOW_X, X, y, words, C, train, test] = [data[s] for s in ['BOW_X', 'X', 'Y', 'words', 'C', 'TR', 'TE']]

        print 'original shapes'
        for x in [BOW_X, X, y, C, train, test, words]:
                print np.shape( x[0] ) 

        [train, test] = [[],[]]

        n = len( C[0] )
        idxs = range( n )
        if random:
                idxs = shuffle( idxs )

        idxs = idxs[: n/scale]
        print 'indices ', len( idxs )
        [BOW_X, X, y, words, C] = [ [ x[0][idxs] ] for x in [BOW_X, X, y, words, C] ]

        train = [[[range( 1, 9*len( idxs ) / 10 )]]]
        test  = [[[range( 1 + 9*len( idxs ) / 10, 1 + len( idxs ) )]]]

        print 'new shapes'
        for x in [BOW_X, X, y, C, train, test, words]:
                print np.shape( x[0] ) 

        sio.savemat(fname + '_sampled_' + str(scale), mdict={'X': X, 'BOW_X': BOW_X, 'Y': y, 'C': C, 'words': words, 'TR': train, 'TE':test})

class lin_transform_wmd():

	def load_data(self, fname):
		try:
			data = sio.loadmat(fname)
			[self.BOW_X, self.y, self.words] = [data[s][0] for s in ['BOW_X', 'Y', 'words']]
                        [self.train, self.test] = [data[s] for s in ['TR','TE']]

                        data = []
                        print 'data loaded'

                        for x in [self.BOW_X, self.y, self.train, self.test, self.words]:
                                print np.shape( x ) 
                

                        n = len( self.BOW_X )
                        for i in xrange(n):
                                
                                bow_i = self.BOW_X[i]
                                bow_i = bow_i / np.sum(bow_i)
                                bow_i = bow_i.tolist()
                                self.BOW_X[i] = bow_i

                                if not i % 100:
                                        print 'completed ', i, ' of ',n

                        for x in [self.BOW_X, self.y, self.train, self.test, self.words]:
                                print np.shape( x ) 


		except Exception as e:
			print e
			raise e
			
	def sample_for_sgd(self):
		true_sample = [ sample(self.class2id_d[yid],2) for yid in sample( set(self.tr_y), self.true_sample_size ) ]
		self.true_pairs = [(m1,m2) for tsp in true_sample for m1,m2 in itertools.combinations(tsp,2) ]
		false_sample = [ (f1,f2) for f1,f2 in zip( sample(range( len(self.tr_y) ),self.false_sample_size*2), sample(range( len(self.tr_y) ),self.false_sample_size*2) ) if self.tr_y[f1] != self.tr_y[f2] ]
		self.false_pairs = sample(false_sample, self.false_sample_size)
                print 'SGD sampled ', len( self.true_pairs ), len( self.false_pairs )


	def set_current_words_distance_matrix(self):
		current_ids = [ idx for l in self.true_pairs + self.false_pairs for idx in l ]
		current_words = set( [word for idx in current_ids for w in self.tr_words[idx] for word in w[0] ] )
		self.words_map = { w:idx for w,idx in enumerate(current_words) }
		self.reverse_words_map = { idx:w for idx,w in enumerate(current_words) }
		self.cw_dist_mat = np.zeros( (len(current_words), len(current_words)) ) 
		for i1,w1 in enumerate(current_words):
			for i2,w2 in enumerate(current_words):
				if i2 > i1:
                                    #print w1
                                    self.cw_dist_mat[i1][i2] = self.A_dist( w1, w2 )
                                    if i2*i1 % 10 == 0:
                                        print 'words distance ', i2*i1
		self.cw_dist_mat += self.cw_dist_mat.T 
		
	def A_dist(self,word1,word2):
		w1 = np.array( self.word2vec[word1] )
		w2 = np.array( self.word2vec[word2] )
		return np.sqrt( np.sum( (self.A.dot(w1-w2))**2 ) )

	def dist(self,word1,word2):
		w1 = np.array( self.word2vec[word1] )
		w2 = np.array( self.word2vec[word2] )
		return np.sqrt( np.sum( (w1-w2)**2 ) )

        def get_gradient(self):
		grad_A = np.zeros_like( self.A )
		for m1,m2 in self.true_pairs:
                        grad_A -= self.alpha * self.get_one_grad(m1,m2)
                        print 'grad true ', m1, m2
		for f1,f2 in self.false_pairs:
			grad_A += self.beta * self.get_one_grad(f1,f2)
                        print 'grad false ', f1, f2
                return grad_A

	def get_one_grad(self, id1, id2):
		words1_d = {word[0]:self.BOW_X[id1][0][i] for i,w in enumerate( self.tr_words[id1] ) for word in w}
		words2_d = {word[0]:self.BOW_X[id2][0][i] for i,w in enumerate( self.tr_words[id2] ) for word in w}
		vec1 = [ words1_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.tr_words[id1] else 0. for i in range( len(self.words_map) ) ]
		vec2 = [ words2_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.tr_words[id2] else 0. for i in range( len(self.words_map) ) ]
		
		part_grad_A = np.zeros_like( self.A )
		
		T = emd_with_flow( np.array(vec1), np.array(vec2), self.cw_dist_mat )
		for id1,t1 in enumerate(T[1]):
			for id2,t2 in enumerate(t1):
				if t2 > 0.:
					x = np.array(self.word2vec[ self.reverse_words_map[id1] ]) - np.array(self.word2vec[ self.reverse_words_map[id2] ])
					const = t2 / (2 * np.sqrt( np.sum( self.A.dot(x)**2 ) )) if np.sum( x**2 ) != 0 else 0
					for i in range(self.vector_dim):
						Ax = self.A[i,:].dot(x)
						for j in range(self.vector_dim):
							part_grad_A[i,j] += const * 2 * Ax * x[j]
		return part_grad_A

		
	# def get_gradient(self):
	# 	self.grad_A = np.zeros_like( self.A )

        #         pool = mp.Pool(processes=self.num_procs)

        #         pool_outputs = pool.map(global_mp_grads_minus, zip( [self]*len(self.true_pairs), self.true_pairs ))
        #         pool.close()
        #         pool.join()
                
        #         pool = mp.Pool(processes=self.num_procs)
        #         # for i in xrange( len( self.true_pairs ) ):
        #         #         grad_A -= pool_outputs[i]

        #         pool_outputs = pool.map(global_mp_grads_plus, zip( [self]*len(self.false_pairs), self.false_pairs ))
        #         pool.close()
        #         pool.join()
                
        #         # for i in xrange( len( self.false_pairs ) ):
        #         #         grad_A += pool_outputs[i]

        #         # return grad_A

	# def set_one_grad_minus(self, ids):
        #         [id1,id2] = ids
	# 	words1_d = {word[0]:self.BOW_X[id1][0][i] for i,w in enumerate( self.tr_words[id1] ) for word in w}
	# 	words2_d = {word[0]:self.BOW_X[id2][0][i] for i,w in enumerate( self.tr_words[id2] ) for word in w}
	# 	vec1 = [ words1_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.tr_words[id1] else 0. for i in range( len(self.words_map) ) ]
	# 	vec2 = [ words2_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.tr_words[id2] else 0. for i in range( len(self.words_map) ) ]
		
	# 	part_grad_A = np.zeros_like( self.A )
		
	# 	T = emd_with_flow( np.array(vec1), np.array(vec2), self.cw_dist_mat )
	# 	for id1,t1 in enumerate(T[1]):
	# 		for id2,t2 in enumerate(t1):
	# 			if t2 > 0.:
	# 				x = np.array(self.word2vec[ self.reverse_words_map[id1] ]) - np.array(self.word2vec[ self.reverse_words_map[id2] ])
	# 				const = t2 / (2 * np.sqrt( np.sum( self.A.dot(x)**2 ) )) if np.sum( x**2 ) != 0 else 0
	# 				for i in range(self.vector_dim):
	# 					Ax = self.A[i,:].dot(x)
	# 					for j in range(self.vector_dim):
	# 						part_grad_A[i,j] += const * 2 * Ax * x[j]
        #         self.grad_A -= part_grad_A

	# def set_one_grad_plus(self, ids):
        #         [id1,id2] = ids
	# 	words1_d = {word[0]:self.BOW_X[id1][0][i] for i,w in enumerate( self.tr_words[id1] ) for word in w}
	# 	words2_d = {word[0]:self.BOW_X[id2][0][i] for i,w in enumerate( self.tr_words[id2] ) for word in w}
	# 	vec1 = [ words1_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.tr_words[id1] else 0. for i in range( len(self.words_map) ) ]
	# 	vec2 = [ words2_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.tr_words[id2] else 0. for i in range( len(self.words_map) ) ]
		
	# 	part_grad_A = np.zeros_like( self.A )
		
	# 	T = emd_with_flow( np.array(vec1), np.array(vec2), self.cw_dist_mat )
	# 	for id1,t1 in enumerate(T[1]):
	# 		for id2,t2 in enumerate(t1):
	# 			if t2 > 0.:
	# 				x = np.array(self.word2vec[ self.reverse_words_map[id1] ]) - np.array(self.word2vec[ self.reverse_words_map[id2] ])
	# 				const = t2 / (2 * np.sqrt( np.sum( self.A.dot(x)**2 ) )) if np.sum( x**2 ) != 0 else 0
	# 				for i in range(self.vector_dim):
	# 					Ax = self.A[i,:].dot(x)
	# 					for j in range(self.vector_dim):
	# 						part_grad_A[i,j] += const * 2 * Ax * x[j]
        #         self.grad_A += part_grad_A


	def train_init(self):
                print 'training...'
		for split in xrange(1):# len(self.train) ):
                        with open( 'log.txt', 'a' ) as f:
                                f.write( str(split) + '\n' )
                        print split

                        self.fold = split

			#load_data(fname)
			train_ix = self.train[split][0][0] - 1
			test_ix = self.test[split][0][0] - 1
                        #test_ix = sample( test_ix, len( test_ix ) / 3 )

                        #print train_ix[:10]
			[self.te_BOW_X, self.te_y, self.te_words] = [self.BOW_X[test_ix], self.y[test_ix], self.words[train_ix] ]
			[self.tr_BOW_X, self.tr_y, self.tr_words] = [self.BOW_X[train_ix], self.y[train_ix], self.words[train_ix] ]


                        #print self.te_y
			print len( set( self.tr_y ) ), len( set( self.te_y ) )
                        self.A = np.identity( self.vector_dim )
                        print 'A initialized'

			self.class2id_d = { yi: [] for yi in set(self.tr_y) }
			for i,yi in enumerate(self.tr_y):
				self.class2id_d[yi].append(i)	
                        print 'class2id size ', len( self.class2id_d )

			for iteration in xrange(self.iterations):
				if iteration % 10 == 0:
					print 'training split ', split, ' iteration ', iteration
				
				self.sample_for_sgd()
				self.set_current_words_distance_matrix()
				#self.get_gradient()
				self.A += self.get_gradient()
                        
			print split, ' trained'
			# self.test_trained()
                        # self.save_A( 'results_Amat', split )

        def save_A( self, path, split ):
                with open( path + str( split ), 'w' ) as f:
                        pickle.dump( self.A, f, -1 )
	
        def mp_wmd( self, id1 ):
                n = len( self.te_y )
                wmd_A_dm = np.zeros((1,n))
                wmd_dm   = np.zeros((1,n))

                words1_d = {word[0]:self.BOW_X[id1][0][i] for i,w in enumerate( self.te_words[id1] ) for word in w}
                vec1 = [ words1_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.te_words[id1] else 0. for i in range( len(self.words_map) ) ]

                print 'wmd/wmd-A for ', id1, ' of ', n
                for id2 in range( id1 ):
                        words2_d = {word[0]:self.BOW_X[id2][0][i] for i,w in enumerate( self.te_words[id2] ) for word in w}
                        vec2 = [ words2_d[self.reverse_words_map[i]] if self.reverse_words_map[i] in self.te_words[id2] else 0. for i in range( len(self.words_map) ) ]


                        wmd_A_dm[0,id2] = emd( np.array(vec1), np.array(vec2), self.cw_dist_mat )
                        wmd_dm[0,id2] = emd( np.array(vec1), np.array(vec2), self.cw_dist_mat_non_A )
                return [wmd_A_dm,wmd_dm]

	def test_trained(self):
		self.testing_words_distance_matrix()
		self.te_class2id_d = { yi: [] for yi in set(self.te_y) }
		for i,yi in enumerate(self.te_y):
                        self.te_class2id_d[yi].append(i)
		
		n = len( self.te_y )
		wmd_A_dm = np.zeros( (n,n) )
		wmd_dm = np.zeros( (n,n) )
		
                print 'length : ', len( self.te_y )

                pool = mp.Pool(processes=self.num_procs)

                pool_outputs = pool.map(global_mp_wmd, zip( [self]*n, list(range(n)) ))
                pool.close()
                pool.join()
                
                for i in xrange(n):
                        wmd_A_dm[:,i] = pool_outputs[i][0]
                        wmd_dm[:,i] = pool_outputs[i][1]

		wmd_A_dm += wmd_A_dm.T
		wmd_dm += wmd_dm.T
		
		wmd_A_true = []
		wmd_orig_t = []
		
		for k in self.te_class2id_d:
			id1 = self.te_class2id_d[k][0]
			id1_matches = len( self.te_class2id_d[k] )

                        with open( 'log.txt', 'a' ) as f:
                                f.write( str(self.te_y[id1]) + ':' + str(id1_matches) + ';' + 
                                         str( [ (self.te_y[idx],wmd_A_dm[id1,idx]) for idx in np.argsort( wmd_A_dm[id1] )[:id1_matches + 5] ]) +  ';' + 
                                         str( [ (self.te_y[idx],wmd_dm)            for idx in np.argsort( wmd_dm[id1] )[:id1_matches + 5] ]) + '\n' )

			wmd_A_true += [ self.te_y[idx] == self.te_y[id1] for idx in np.argsort( wmd_A_dm[id1] )[1:id1_matches] ]
			wmd_orig_t += [ self.te_y[idx] == self.te_y[id1] for idx in np.argsort( wmd_dm[id1] )[1:id1_matches] ]
			
		print 'trained WMD top k acc   : ', sum( wmd_A_true ) / float( len(wmd_A_true) )
		print 'untrained WMD top k acc : ', sum( wmd_orig_t ) / float( len(wmd_orig_t) )
                with open( 'log.txt', 'a' ) as f:
                        f.write( 'trained WMD top k acc   : ' + str( sum( wmd_A_true ) / float( len(wmd_A_true) ) ) + '\n' )
                        f.write( 'untrained WMD top k acc : ' + str( sum( wmd_orig_t ) / float( len(wmd_orig_t) ) ) + '\n' )

                save_file = 'results/dists_4-17'
                with open(save_file + '_learned_A-WMD_' + str(self.fold), 'w') as f:
                        pickle.dump([[], self.te_y, wmd_A_dm], f)

                with open(save_file + '_learned_non_A-WMD_' + str(self.fold), 'w') as f:
                        pickle.dump([[], self.te_y, wmd_dm], f)

                

	def testing_words_distance_matrix(self):

                current_words = set( [word for w in self.te_words for word in w[0][0] ] )

		self.words_map = { w:idx for w,idx in enumerate(current_words) }
		self.reverse_words_map = { idx:w for idx,w in enumerate(current_words) }
		self.cw_dist_mat = np.zeros( (len(current_words), len(current_words)) )
		self.cw_dist_mat_non_A = np.zeros( (len(current_words), len(current_words)) )
		for i1,w1 in enumerate(current_words):
			for i2,w2 in enumerate(current_words):
				if i2 > i1:
					self.cw_dist_mat[i1][i2] = self.A_dist( w1, w2 )
					self.cw_dist_mat_non_A[i1][i2] = self.dist( w1, w2 )
		self.cw_dist_mat += self.cw_dist_mat.T 
		self.cw_dist_mat_non_A += self.cw_dist_mat_non_A.T 
	
        def __init__(self, fname, iterations = 2, true_sample_size = 2, false_sample_size = 2, alpha = 0.01, beta = 0.01, vector_dim = 300):
                self.fname = fname
		self.iterations = iterations
		self.true_sample_size = true_sample_size
		self.false_sample_size = false_sample_size
		self.alpha = alpha
		self.beta = beta
                self.vector_dim = vector_dim

                self.words = []
                self.word2vec = pickle.load( open( 'data/sample_dictionary.pk' ) )
                self.y = []

                self.num_procs = 4

		try:
			self.load_data(fname)
                except Exception as e:
			print e
                # try:
		# 	print 'building vector dictionary...'
		# 	self.build_w2v()
		# 	print 'dictionary complete'
		# except Exception as e:
		# 	print e
			#return False
                print 'y shape', np.shape( self.y )

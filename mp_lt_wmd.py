#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#lt = ltw.lin_transform_wmd( 'data/OT_reviews_sampled_10.mat' )
"""
Created on Tue April 17 2018

@author: a.schneider
"""

import numpy as np
from random import sample, shuffle
import itertools
import sys
import time
import pickle
import scipy.io as sio
from pyemd import emd_with_flow
from functools import partial
sys.path.append('python-emd-master')
from emd import emd



import multiprocessing as mp
num_procs = 2

logfile = 'amazon-7-5-2018'


def wmd_distance(x1,x2):
    return np.sqrt( np.sum((np.array(x1) - np.array(x2))**2) )

def A_wmd_distance(A,x1,x2):
    return np.sqrt( np.sum((A.dot( np.array(x1) ) - A.dot( np.array(x2) ))**2) )


def global_mp_unique_word_dists( arg, **kwarg ):
    return lin_transform_wmd.unique_word_dists( *arg, **kwarg )

def global_mp_wmd( arg, **kwarg ):
    return lin_transform_wmd.mp_wmd( *arg, **kwarg )


class lin_transform_wmd():
    def load_data(self, fname, split):
        [self.te_BOW_X, self.te_X, self.te_y, self.te_words] = [ [] for _ in range(4) ]
        [self.tr_BOW_X, self.tr_X, self.tr_y, self.tr_words] = [ [] for _ in range(4) ]
        data = {}
        try:
            # data = sio.loadmat(fname)
            sio.loadmat(fname, mdict=data)
            print data.keys()
            [BOW_X, y, X, words] = [data[s][0] for s in ['BOW_X', 'y', 'X', 'indices' ]]#'words']]
            # [BOW_X, y, X, words] = [data[s][0] for s in ['BOW_X', 'y', 'X', 'indices' ]]#'words']]
            # [train, test] = [data[s][0] for s in ['TR','TE']]
            [train, test] = [data[s] for s in ['TR','TE']]
            
            data = []
            print 'data loaded'
            print [np.shape(train), np.shape(test)]
            # print [train[0], test[0]]
            print 'BOW_X shape ', np.shape(BOW_X)
            print 'y shape ', np.shape(y)

        except Exception as e:
            print e
            raise e

        train_ix = train[split] - 1
        test_ix =  test[split] - 1
        # train_ix = train[split][0] - 1
        # test_ix =  test[split][0] - 1
        # print 'test ix ', test_ix
        if len( test_ix ) > self.test_sample_size:
            test_ix = sample( test_ix, self.test_sample_size )
            
        #print train_ix[:10]
        [self.te_BOW_X, self.te_X, self.te_y, self.te_words] = [BOW_X[test_ix], X[test_ix], y[test_ix], words[test_ix] ]
        [self.tr_BOW_X, self.tr_X, self.tr_y, self.tr_words] = [BOW_X[train_ix], X[train_ix], y[train_ix], words[train_ix] ]
            
        print 'type_ ', [type(BOW_X), type(y)]

        print 'indices ', [np.shape(train_ix), np.shape(test_ix)]#, [train_ix, test_ix]

        print 'te_ ', [np.shape(self.te_BOW_X), np.shape(self.te_y)]
        print 'tr_ ', [np.shape(self.tr_BOW_X), np.shape(self.tr_y)]

        print 'train size ', np.shape( self.tr_y )
        print 'test size  ', np.shape( self.te_y )

        n = np.shape(self.tr_X)
        n = n[0]
            
        print 'normalizing X, bow ...'
        for i in xrange(n):
            X_i = self.tr_X[i].T
            # X_i = X_i.tolist()
            self.tr_X[i] = X_i
            bow_i = self.tr_BOW_X[i]
            bow_i = bow_i / float(np.sum(bow_i))
            bow_i = bow_i.tolist()
            self.tr_BOW_X[i] = bow_i

            # if i % 100 == 0:
            #     print 'X ', i, ' of ', n
            #     print bow_i

        print 'Xs train initialized'

        n = np.shape(self.te_X)
        n = n[0]
            
        for i in xrange(n):
            X_i = self.te_X[i].T
            # X_i = X_i.tolist()
            self.te_X[i] = X_i
            bow_i = self.te_BOW_X[i]
            bow_i = bow_i / float(np.sum(bow_i))
            bow_i = bow_i.tolist()
            self.te_BOW_X[i] = bow_i

        print 'Xs test initialized'
        print 'data loaded successfully'
			
    def sample_for_sgd(self):
        # true_sample = [ sample(self.class2id_d[yid],2) for yid in sample( set(self.tr_y), min( len( set(self.tr_y) ), self.true_sample_size ) ) ]
        # self.true_pairs = [(m1,m2) for tsp in true_sample for m1,m2 in itertools.combinations(tsp,2) ]

        self.true_pairs = []
        for yid in set( self.tr_y ):
            if len( self.class2id_d[yid] ) > 2:
                # print yid, self.class2id_d[yid], sample( self.class2id_d[yid], len( self.class2id_d[yid] ) )
                sames = sample( self.class2id_d[yid], len( self.class2id_d[yid] ) )
                for s in sames[1:]:
                    self.true_pairs.append( (sames[0], s) )

        self.true_pairs = sample( self.true_pairs, self.true_sample_size )
	
	# false_sample = [ (f1,f2) for f1,f2 in zip( sample(range( len(self.tr_y) ), min( len(self.tr_y), self.false_sample_size*5 ) ), 
        #                                            sample(range( len(self.tr_y) ), min( len(self.tr_y), self.false_sample_size*5 ) ) ) if self.tr_y[f1] != self.tr_y[f2] ]
        # self.false_pairs = sample(false_sample, self.false_sample_size)

        self.false_pairs = []
        while len( self.false_pairs ) < self.false_sample_size:
            (f1,f2) = sample(range( len(self.tr_y) ), 2 )
            while self.tr_y[f1] == self.tr_y[f2]:
                (f1,f2) = sample(range( len(self.tr_y) ), 2 )
            self.false_pairs.append( (f1,f2) )

        # print 'SGD sampled ', self.true_pairs, self.false_pairs
        print 'SGD sampled ', len( self.true_pairs ), len( self.false_pairs )
        current_words = []
        for m1, m2 in self.true_pairs:
            for w in self.tr_words[m1][0]:
                current_words.append( w )
            for w in self.tr_words[m2][0]:
                current_words.append( w )
        for m1, m2 in self.false_pairs:
            for w in self.tr_words[m1][0]:
                current_words.append( w )
            for w in self.tr_words[m2][0]:
                current_words.append( w )
        # for m1, m2 in self.true_pairs:
        #     for w in self.tr_words[m1][0]:
        #         current_words.append( w[0] )
        #     for w in self.tr_words[m2][0]:
        #         current_words.append( w[0] )
        # for m1, m2 in self.false_pairs:
        #     for w in self.tr_words[m1][0]:
        #         current_words.append( w[0] )
        #     for w in self.tr_words[m2][0]:
        #         current_words.append( w[0] )
        self.current_words = set( current_words )
        print 'current words ', list( self.current_words )[:10], ' ... '
        print 'SGD current words vocab size ', len( self.current_words )
        
    def set_current_words_distance_matrix(self):            
        self.all_words = []
        self.all_X = []
        self.word_id_map = {}

        for i1,_ in enumerate( self.tr_words ):
            # print 'words_i1 ', self.tr_words[i1]
            for i2,_ in enumerate( self.tr_words[i1][0] ):
                # print self.tr_words[i1][0][i2]
                if self.tr_words[i1][0][i2] in self.current_words and self.tr_words[i1][0][i2] not in self.all_words:
                    self.all_words.append( self.tr_words[i1][0][i2] )
                    self.all_X.append( self.tr_X[i1][i2] )

        vocab_size = len( self.all_words )
        print 'vocab size ', vocab_size
        self.word_id_map = { word : k for k,word in enumerate( self.all_words ) }

        # pool = mp.Pool(processes=self.num_procs)
                
        # pool_outputs = pool.map(global_mp_unique_word_dists, zip( [self]*vocab_size, list(range(vocab_size)) ))
        # pool.close()
        # pool.join()

        self.cw_dist_mat = np.zeros((vocab_size,vocab_size))
        for i in xrange(vocab_size):
            self.cw_dist_mat[:,i] = self.unique_word_dists(i)
            # self.cw_dist_mat[:,i] = pool_outputs[i]

        self.cw_dist_mat += self.cw_dist_mat.T

    def unique_word_dists(self, ix):
        n = len(self.all_words)
        Di = np.zeros((1,n))
        i = ix
        if i % 500 == 0:
            print 'unique words %d out of %d' % (i, n)

        for j in xrange(i):
            Di[0,j] = self.A_dist( self.all_X[i], self.all_X[j] )
        return Di
		
    def A_dist(self,vec1,vec2):
        w1 = np.array( vec1 )
        w2 = np.array( vec2 )
        return np.sqrt( np.sum( (self.A.dot(w1-w2))**2 ) )

    # def dist(self,vec1,vec2):
    #     w1 = np.array( vec1 )
    #     w2 = np.array( vec2 )
    #     return np.sqrt( np.sum( (w1-w2)**2 ) )

    def get_gradient(self):
        grad_A = np.zeros_like( self.A )
        count = 0
        for m1,m2 in self.true_pairs:
            grad_A -= self.alpha * self.get_one_grad(m1,m2)
            print count, 'grad true ', m1, m2
            count += 1
        for f1,f2 in self.false_pairs:
            grad_A += self.beta * self.get_one_grad(f1,f2)
            print count, 'grad false ', f1, f2
            count += 1
        return grad_A

    def get_one_grad(self, id1, id2):
        vn = len( self.all_words )
        n = len(self.tr_X)

        vec1 = np.zeros(vn)
        for k,w in enumerate( self.tr_words[id1][0] ):
            word = w#w[0]            
            vec1[ self.word_id_map[word] ] = self.tr_BOW_X[id1][0][k]

        vec2 = np.zeros(vn)
        for k,w in enumerate( self.tr_words[id2][0] ):
            word = w#w[0]
            vec2[ self.word_id_map[word] ] = self.tr_BOW_X[id2][0][k]
        part_grad_A = np.zeros_like( self.A )
        
        T = emd_with_flow( np.array(vec1), np.array(vec2), self.cw_dist_mat )
        for id1,t1 in enumerate(T[1]):
            for id2,t2 in enumerate(t1):
                if t2 > 0.:
                    x = np.array(self.all_X[ id1 ]) - np.array(self.all_X[ id2 ])
                    const = t2 / (2 * np.sqrt( np.sum( self.A.dot(x)**2 ) )) if np.sum( x**2 ) != 0 else 0
                    for i in range(self.vector_dim):
                        Ax = self.A[i,:].dot(x)
                        for j in range(self.vector_dim):
                            part_grad_A[i,j] += const * 2 * Ax * x[j]
        return part_grad_A


    def learn_transform(self):
        print 'training...'
        for split in xrange( 5 ):
            with open( logfile + '.log', 'a' ) as f:
                f.write( str(split) + '\n' )
            print split
            self.fold = split
                    
            try:
                self.load_data(self.fname,split)
            except Exception as e:
                print e

            #print self.te_y
            print 'train test sizes : ', len( set( self.tr_y ) ), len( set( self.te_y ) )
            self.A = np.identity( self.vector_dim )
            print 'A initialized'

            self.class2id_d = { yi: [] for yi in set(self.tr_y) }
            for i,yi in enumerate(self.tr_y):
                self.class2id_d[yi].append(i)	
            print 'class2id size ', len( self.class2id_d )

            for iteration in xrange(self.iterations):
                if True: # iteration % 10 == 0:
                    print 'training split ', split, ' iteration ', iteration
				
                self.sample_for_sgd()
                self.set_current_words_distance_matrix()
                #self.get_gradient()
                self.A += self.get_gradient()
                        
            print split, ' trained'
            self.save_A( '6-24-18_results_A_mat', split )

            self.test_trained()            

    def save_A( self, path, split ):
        with open( path + str( split ), 'w' ) as f:
            pickle.dump( self.A, f, -1 )

    def mp_wmd( self, id1 ):
        n = len( self.te_words )
        A_dist = np.zeros( (1,n) )
        dist = np.zeros( (1,n) )
        
        # id1words = [ w for w in self.te_words[id1][0] ]

        # local_words = [ w for w in self.te_words[id1][0] ]
        # # local_words = [ w[0] for w in self.te_words[id1][0] ]
        # local_X = [ x for x in self.te_X[id1] ]


        print 'wmd/wmd-A for ', id1, ' of ', n

        #AX = A.dot( self.te_X[id1]
        for id2 in range( id1 ):
            # print np.shape( self.te_X[id1].tolist() ), np.shape( self.te_BOW_X[id1][0] )
            AWD = partial( A_wmd_distance, self.A )

            A_dist[0,id2] = emd( (self.te_X[id1].tolist(), self.te_BOW_X[id1][0]), (self.te_X[id2].tolist(), self.te_BOW_X[id2][0]), AWD)
            dist[0,id2] = emd( (self.te_X[id1].tolist(), self.te_BOW_X[id1][0]), (self.te_X[id2].tolist(), self.te_BOW_X[id2][0]), wmd_distance)
        #     local_words = [ w for w in self.te_words[id1][0] ]
        #     # local_words = [ w[0] for w in self.te_words[id1][0] ]
        #     local_X = [ x for x in self.te_X[id1] ]

        #     expansion = 0
        #     id2words = [ w for w in self.te_words[id2][0] ]
        #     # id2words = [ w[0] for w in self.te_words[id2][0] ]
        #     for k,word in enumerate( id2words ):
        #         if word not in local_words:
        #             local_words.append( word )
        #             local_X.append( self.te_X[id2][k] )
        #             expansion += 1

        #     # cw_A_dist = np.pad( cw_A_dist, (0,expansion), 'constant', constant_values=(0))
        #     # cw_non_A_d = np.pad( cw_non_A_d, (0,expansion), 'constant', constant_values=(0))        

        #     cw_A_dist   = np.zeros( (len( local_words ),len( local_words )) )
        #     cw_non_A_d  = np.zeros( (len( local_words ),len( local_words )) )


        #     words_map = { word : k for k,word in enumerate( local_words ) }
            
        #     for w1 in id1words:
        #         for w2 in id2words:
        #             if cw_A_dist[ words_map[w1] ][ words_map[w2] ] == 0.:
        #                 a_dist = self.A_dist( local_X[ words_map[w1] ], local_X[ words_map[w2] ] )
        #                 # print w1, w2, a_dist
        #                 cw_A_dist[ words_map[w1] ][ words_map[w2] ] = a_dist
        #                 cw_A_dist[ words_map[w2] ][ words_map[w1] ] = a_dist

        #                 c_dist = self.dist( local_X[ words_map[w1] ], local_X[ words_map[w2] ] )
        #                 # print w1, w2, c_dist
        #                 cw_non_A_d[ words_map[w1] ][ words_map[w2] ] = c_dist
        #                 cw_non_A_d[ words_map[w2] ][ words_map[w1] ] = c_dist

            
        #     vn = len( local_words )
                        
        #     vec1 = np.zeros(vn)
        #     vec2 = np.zeros(vn)
    
        #     for k,word in enumerate( id1words ):
        #         # vec1[ words_map[word] ]
        #         # print k
        #         vec1[ words_map[word] ] = self.te_BOW_X[id1][0][k]
        #     for k,word in enumerate( id2words ):
        #         vec2[ words_map[word] ] = self.te_BOW_X[id2][0][k]

        #     A_dist[0,id2] = emd( vec1, vec2, cw_A_dist )
        #     dist[0,id2] = emd( vec1, vec2, cw_non_A_d )

        #     if (id2 + id1) % 20 == 0:
        #         print vn, 'A dist : ', A_dist[0,id2], 'NAdist : ', dist[0,id2]
        #         print id1, np.shape( cw_A_dist ), np.count_nonzero( cw_A_dist )
        #         # print vn, 'A dist : ', A_dist[0,id2], np.sum( self.te_BOW_X[id1][0] ), np.sum( vec1 )
        #         # print vn, 'NAdist : ', dist[0,id2], np.sum( self.te_BOW_X[id2][0] ), np.sum( vec2 )

        return [A_dist, dist]

    def test_trained(self):
        self.te_class2id_d = { yi: [] for yi in set(self.te_y) }
        for i,yi in enumerate(self.te_y):
            self.te_class2id_d[yi].append(i)
		
        n = len( self.te_y )
        wmd_A_dm = np.zeros( (n,n) )
        wmd_dm = np.zeros( (n,n) )
        
        print 'length : ', len( self.te_y )
        
        # pool = mp.Pool(processes=self.num_procs)
        # # pool = mp.Pool(processes=1)
        
        # pool_outputs = pool.map(global_mp_wmd, zip( [self]*n, list(range(n)) ))
        # pool.close()
        # pool.join()
        
        # for i in xrange(n):
        #     wmd_A_dm[:,i] = pool_outputs[i][0]
        #     wmd_dm[:,i] = pool_outputs[i][1]

        for i in xrange(n):
            p_outs = self.mp_wmd(i)
            wmd_A_dm[:,i] = p_outs[0]
            wmd_dm[:,i] = p_outs[1]

        wmd_A_dm += wmd_A_dm.T
        wmd_dm += wmd_dm.T
		
        wmd_A_true = []
        wmd_orig_t = []
		


        for k in self.te_class2id_d:
            id1 = self.te_class2id_d[k][0]
            id1_matches = len( self.te_class2id_d[k] )


            with open( logfile + '_cf.log', 'a' ) as f:
                f.write( '**********\n' )
                f.write( 'WMD_A\n' )
                f.write( str(self.te_y[id1]) + ':' + str(id1_matches) + ';' + 
                         str([ (self.te_y[idx],wmd_A_dm[id1,idx]) for idx in np.argsort( wmd_A_dm[id1] )[:id1_matches + 5] ]) + '\n' )
                f.write( str( id1 ) + ' matches ' + str( [ self.te_y[idx] for idx in np.argsort( wmd_A_dm[id1] )[1:id1_matches] ] ) )
                f.write( 'trues ' + str( [ self.te_y[idx] == self.te_y[id1] for idx in np.argsort( wmd_A_dm[id1] )[1:id1_matches] ] ) )
                f.write( ' of ' + str( id1_matches ) )
                f.write( '\n**********\n' )
                f.write( 'WMD_NA\n' )
                f.write( str(self.te_y[id1]) + ':' + str(id1_matches) + ';' + 
                         str([ (self.te_y[idx],wmd_dm[id1,idx]) for idx in np.argsort( wmd_dm[id1] )[:id1_matches + 5] ]) + '\n' )
                f.write( str( id1 ) + ' matches ' + str( [ self.te_y[idx] for idx in np.argsort( wmd_dm[id1] )[1:id1_matches] ] ) )
                f.write( 'trues ' + str( [ self.te_y[idx] == self.te_y[id1] for idx in np.argsort( wmd_dm[id1] )[1:id1_matches] ] ) )
                f.write( ' of ' + str( id1_matches ) + '\n' )

            with open( logfile + '.log', 'a' ) as f:
                f.write( str(self.te_y[id1]) + ':' + str(id1_matches) + ';' + 
                         str( [ (self.te_y[idx],wmd_A_dm[id1,idx]) for idx in np.argsort( wmd_A_dm[id1] )[:id1_matches + 5] ]) +  ';' + 
                         str( [ (self.te_y[idx],wmd_dm[id1,idx])   for idx in np.argsort( wmd_dm[id1] )[:id1_matches + 5] ]) + '\n' )

            wmd_A_true += [ self.te_y[idx] == self.te_y[id1] for idx in np.argsort( wmd_A_dm[id1] )[1:id1_matches] ]
            wmd_orig_t += [ self.te_y[idx] == self.te_y[id1] for idx in np.argsort( wmd_dm[id1] )[1:id1_matches] ]
			
        print 'trained WMD top k acc   : ', np.mean( wmd_A_true )
        print 'untrained WMD top k acc : ', np.mean( wmd_orig_t ) 
        with open( logfile + '.log', 'a' ) as f:
            f.write( 'trained WMD top k acc   : ' + str( np.mean( wmd_A_true ) ) + '\n' )
            f.write( 'untrained WMD top k acc : ' + str( np.mean( wmd_orig_t ) ) + '\n' )

        save_file = 'results/' + logfile
        with open(save_file + '_learned_A-WMD_' + str(self.fold), 'w') as f:
            pickle.dump([[], self.te_y, wmd_A_dm], f)

        with open(save_file + '_learned_non_A-WMD_' + str(self.fold), 'w') as f:
            pickle.dump([[], self.te_y, wmd_dm], f)

                

    # def testing_words_distance_matrix(self):
    #     current_words = set( [word for w in self.te_words for word in w[0][0] ] )
        
    #     self.words_map = { w:idx for w,idx in enumerate(current_words) }
    #     self.reverse_words_map = { idx:w for idx,w in enumerate(current_words) }
    #     self.cw_dist_mat = np.zeros( (len(current_words), len(current_words)) )
    #     self.cw_dist_mat_non_A = np.zeros( (len(current_words), len(current_words)) )
    #     for i1,w1 in enumerate(current_words):
    #         for i2,w2 in enumerate(current_words):
    #             if i2 > i1:
    #                 self.cw_dist_mat[i1][i2] = self.A_dist( w1, w2 )
    #                 self.cw_dist_mat_non_A[i1][i2] = self.dist( w1, w2 )
    #     self.cw_dist_mat += self.cw_dist_mat.T 
    #     self.cw_dist_mat_non_A += self.cw_dist_mat_non_A.T 
	
    def __init__(self, fname, iterations = 30, true_sample_size = 10, false_sample_size = 10, alpha = 0.2, beta = 0.1, vector_dim = 300, num_cores=num_procs, test_sample_size = 100):
        self.fname = fname
        self.iterations = iterations
        self.true_sample_size = true_sample_size
        self.false_sample_size = false_sample_size
        self.alpha = alpha
        self.beta = beta
        self.vector_dim = vector_dim
        self.test_sample_size = test_sample_size
        self.words = []
        # self.word2vec = pickle.load( open( 'data/sample_dictionary.pk' ) )
        self.y = []

        self.num_procs = num_cores

def sample_data( fname, scale, random=False ):
    data = sio.loadmat(fname)
    [BOW_X, X, y, words, C, train, test] = [data[s] for s in ['BOW_X', 'X', 'y', 'words', 'C', 'TR', 'TE']]
    
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
        
    sio.savemat(fname + '_sampled_' + str(scale), mdict={'X': X, 'BOW_X': BOW_X, 'y': y, 'C': C, 'words': words, 'TR': train, 'TE':test})
    
def main(argv):
    print argv 
    if len( argv ) <= 1:
        input_file = 'data/sampled_20_words_tfidftr_te_split.mat'
    else:
        input_file = argv[1]

    lt = lin_transform_wmd( input_file )
    lt.learn_transform()

if __name__ == "__main__":
    main(sys.argv)

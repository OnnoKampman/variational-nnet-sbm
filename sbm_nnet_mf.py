import os
import shutil
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import numpy as np
import scipy.sparse as sp
from scipy.misc import logsumexp

from utils import BatchGenerator, get_pairs, log_gaussian_density


class SbmNNetMF:

    def __init__(self):
        """
        Base class for a model for binary symmetric link matrices with blockmodel structure.
        """


    def construct_graph(self):

        N = self.N  # number of nodes
        T = self.T  # truncation level of the Dirichlet process
        n_features = self.n_features
        hidden_layer_sizes = self.hidden_layer_sizes


        #######################################################
        ###  List all placeholders here for easy reference  ###
        #######################################################

        self.row = tf.placeholder(dtype=tf.int32, shape=[None])
        self.col = tf.placeholder(dtype=tf.int32, shape=[None])
        self.val = tf.placeholder(dtype=tf.int32, shape=[None])

        self.batch_scale = tf.placeholder(dtype=tf.float32, shape=[], name='batch_scale')
        self.n_samples = tf.placeholder(dtype=tf.int32, shape=[], name='n_samples')

        # The variational parameters for the local assignment variables, Z_i, are analytically updated and passed in
        # as placeholders.
        self.qZ = tf.placeholder(dtype=tf.float32, shape=[N, T], name='qZ')

        self.sum_qZ_above = tf.placeholder(dtype=tf.float32, shape=[N, T - 1], name='sum_qZ_above')


        #############################################
        ###  Create the features and nnet inputs  ###
        #############################################

        init_scale = - 4.6  # initial scale of inv_softplus(sigma), for noise std devs sigma; -4.6 maps to 0.01 under softplus

        # the node-specific features are vectors drawn from a cluster specific distribution
        # make the prior distributions for each cluster
        self.pU_dist = ds.Normal(loc=tf.Variable(tf.random_normal([n_features]), name='pU_mean'),
                                 scale=tf.nn.softplus(tf.Variable(tf.ones([n_features]) * init_scale, name='pU_std_unc')),
                                 name='pU_dist')

        self.qU_dist = ds.Normal(loc=tf.Variable(tf.random_normal([T, n_features]), name='qU_mean'),
                                 scale=tf.nn.softplus(tf.Variable(tf.ones([T, n_features]) * init_scale, name='qU_std_unc')),
                                 name='qU_dist')

        qU_samps = self.qU_dist.sample(self.n_samples)  # (n_samples, T, n_features)


        # We must integrate w.r.t. qZ, which requires an eventual sum over all possible combinations of q(Z_i), q(Z_j),
        # for each (i, j) in the minibatch. But this requires us to compute the likelihood for each possible Z_i, Z_j.
        all_T_pairs = np.concatenate([np.tril_indices(T, k=0),  # includes diagonal
                                      np.triu_indices(T, k=1)
                                      ], axis=1)  # (2, n_T_pairs)

        row_features = tf.gather(qU_samps, indices=all_T_pairs[0], axis=1)  # (n_samples, n_T_pairs, n_features)
        col_features = tf.gather(qU_samps, indices=all_T_pairs[1], axis=1)

        inputs_ = tf.concat([row_features,
                             col_features
                             ], axis=2)  # (n_samples, n_T_pairs, n_inputs)


        ###################################
        ###  Create the neural network  ###
        ###################################

        # all weights share a prior under p
        self.pW_dist = ds.Normal(loc=tf.Variable(tf.random_normal([1]), name='pW_mean'),
                                 scale=tf.nn.softplus(tf.Variable(init_scale, name='pW_std_unc')),
                                 name='pW_dist')

        # the biases are also random variables, which I find necessary for good performance (as opposed to just
        # params of the ELBO)
        self.pB_dist = ds.Normal(loc=tf.Variable(tf.random_normal([1]), name='pB_mean'),
                                 scale=tf.nn.softplus(tf.Variable(init_scale, name='pB_std_unc')),
                                 name='pB_dist')

        # we will collect up the weight and bias tensors for reference later
        self.nnet_tensors = []  # will be a list of (W, b) tuples

        activation_fn = tf.nn.relu
        n_inputs = tf.cast(inputs_.shape[-1], tf.int32)
        for layer_i, layer_size in enumerate(hidden_layer_sizes):  # if an empty list then this loop is not entered
            
            with tf.name_scope("NN_layer_%d" % layer_i):

                # the p and q distributions for this layer's weights
                qW_dist = ds.Normal(loc=tf.Variable(tf.random_normal([n_inputs, layer_size]), name='qW_layer_mean'),
                                    scale=tf.nn.softplus(tf.Variable(tf.ones([n_inputs, layer_size]) * init_scale, name='qW_layer_std_unc')),
                                    name='qW_layer_dist')

                # biases
                qB_dist = ds.Normal(loc=tf.Variable(tf.random_normal([layer_size]), name='qB_mean'),
                                    scale=tf.nn.softplus(tf.Variable(tf.ones([layer_size]) * init_scale, name='qB_std_unc')),
                                    name='qB_dist')

                W_samps = qW_dist.sample(self.n_samples)  # (n_samples, prev_layer_size, layer_size)
                B_samps = qB_dist.sample(self.n_samples)  # (n_samples, layer_size)

                # inputs_ will be (n_samples, n_T_pairs, prev_layer_size)
                inputs_ = activation_fn(tf.matmul(inputs_, W_samps) + B_samps[:, None, :])  # (n_samples, n_T_pairs, layer_size)

                n_inputs = layer_size

                # store the distribution objects, but we won't need the samples
                self.nnet_tensors.append((qW_dist, qB_dist))


        # the output layer mapping to a single probability of a link
        qW_dist = ds.Normal(loc=tf.Variable(tf.random_normal([n_inputs, 1]), name='qW_out_mean'),
                            scale=tf.nn.softplus(tf.Variable(tf.ones([n_inputs, 1]) * init_scale, name='qW_out_std_unc')),
                            name='qW_out_dist')

        qB_dist = ds.Normal(loc=tf.Variable(tf.random_normal([1]), name='qB_out_mean'),
                            scale=tf.nn.softplus(tf.Variable(init_scale, name='qB_out_std_unc')),
                            name='qB_out_dist')

        self.nnet_tensors.append((qW_dist, qB_dist))

        W_samps = qW_dist.sample(self.n_samples)  # (n_samples, prev_layer_size, 1)
        B_samps = qB_dist.sample(self.n_samples)  # (n_samples, 1)... not sure why the 1 trails here in the shape...

        # inputs_ is (n_samples, n_T_pairs, final_layer_size)
        logits = tf.matmul(inputs_, W_samps) + B_samps[:, None, :]  # (n_samples, n_T_pairs, 1)


        ################################
        ###  Compute the likelihood  ###
        ################################

        # cross entropy is negative Bernoulli log-likelihood
        n_T_pairs = all_T_pairs.shape[1]
        batch_size = tf.shape(self.val)[0]
        val_ = tf.tile(self.val[None, None, :], [self.n_samples, n_T_pairs, 1])  # must be the same type and shape as logits
        logits_ = tf.tile(logits, [1, 1, batch_size])  # (n_samples, n_T_pairs, batch_size)
        log_bernoulli_likel = - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(val_, tf.float32),
                                                                        logits=logits_)  # (n_samples, n_T_pairs, batch_size)

        self.log_bernoulli_likel = tf.transpose(log_bernoulli_likel, [0, 2, 1])  # (n_samples, batch_size, n_T_pairs)

        # should check if the following indexing is better done outside TF graph
        qZ_row = tf.gather(self.qZ, indices=self.row, axis=0)  # (batch_size, T)
        qZ_col = tf.gather(self.qZ, indices=self.col, axis=0)
        qZ_pairs_row = tf.gather(qZ_row, indices=all_T_pairs[0], axis=1)  # (batch_size, n_T_pairs)
        qZ_pairs_col = tf.gather(qZ_col, indices=all_T_pairs[1], axis=1)  # (batch_size, n_T_pairs)

        loglikel = tf.einsum('jk,ijk->ij', qZ_pairs_row * qZ_pairs_col, self.log_bernoulli_likel)  # (n_samples, batch_size); presumably faster than tile->matmul


        ##############################
        ###  Compute the KL terms  ###
        ##############################

        kl_divergence = tf.constant(0.0)  # will broadcast up

        # KL terms of U (analytically evaluated)
        kl_divergence += tf.reduce_sum(self.qU_dist.kl_divergence(self.pU_dist))  # scalar

        # nnet weights and biases
        for qW_dist, qB_dist in self.nnet_tensors:  # will have at least one entry
            kl_divergence += tf.reduce_sum(qW_dist.kl_divergence(self.pW_dist))  # scalar
            kl_divergence += tf.reduce_sum(qB_dist.kl_divergence(self.pB_dist))  # scalar

        # KL terms for the DP sticks V; V can be analytically updated but we'll prefer to do gradient updates
        self.dp_conc = tf.nn.softplus(tf.Variable(3.5, name='dp_conc_unc'))  # 3.5 maps to 3.0 under softplus

        self.qV_shp1 = tf.nn.softplus(tf.Variable(tf.ones(T - 1) * 0.54, name='qV_shp1'))  # 0.54 maps to 1.0 under softplus
        self.qV_shp2 = tf.nn.softplus(tf.Variable(tf.ones(T - 1) * 0.54, name='qV_shp2'))

        digamma_sum = tf.digamma(self.qV_shp1 + self.qV_shp2)
        self.E_log_V = tf.digamma(self.qV_shp1) - digamma_sum  # (T-1,)
        self.E_log_1mV = tf.digamma(self.qV_shp2) - digamma_sum

        # KL terms for E[log p(Z|V)] with V integrated out (verified this, it's correct)
        # note KL divergence is E_q [logq / logp]
        kl_divergence += - tf.reduce_sum(self.sum_qZ_above * self.E_log_1mV + self.qZ[:, :-1] * self.E_log_V) \
                            + tf.reduce_sum(self.qZ * tf.log(self.qZ))
        
        # elbo terms for E[log p(V|c)]
        kl_divergence += - tf.log(self.dp_conc) + (self.dp_conc - 1.0) * tf.reduce_sum(self.E_log_1mV) \
                            + tf.reduce_sum( tf.lgamma(self.qV_shp1 + self.qV_shp2) - tf.lgamma(self.qV_shp1) - tf.lgamma(self.qV_shp2)
                                                + (self.qV_shp1 - 1.0) * self.E_log_V + (self.qV_shp2 - 1.0) * self.E_log_1mV
                                            )  # a scalar

        ###########################
        ###  Assemble the ELBO  ###
        ###########################

        self.data_loglikel = tf.reduce_sum(loglikel) / tf.cast(self.n_samples, tf.float32)  # will be recorded
        self.elbo = self.batch_scale * self.data_loglikel - kl_divergence



    def train(self, N, row, col, T, n_features, hidden_layer_sizes, n_iterations, batch_size,
              n_samples, holdout_ratio, learning_rate, root_savedir, root_logdir, no_train_metric=False, seed=None, debug=False):

        """
        Training routine.

        Note about the data: the (row, col) tuples of the ON (i.e., one-valued) entries of the graph are to be passed,
        and they should correspond to the upper triangle of the graph. (Recall we do not allow self-links.) Regardless,
        the code will make a symmetric graph out of all passed entries (within the upper triangular or not) and only the
        upper triangle of the resulting matrix will be kept.

        :param N: Number of nodes in the graph.
        :param row: row indices corresponding to the ON entries (in the upper triangle).
        :param col: col indices corresponding to the ON entries (in the upper triangle).
        :param T: Truncation level for the DP.
        :param n_features:
        :param hidden_layer_sizes:
        :param n_iterations:
        :param batch_size: HALF the minibatch size. In particular, we will always add the symmetric entry in the graph
            (i.e., the corresponding entry in the lower triangle) in the minibatch.
        :param n_samples:
        :param holdout_ratio:
        :param learning_rate:
        :param root_savedir:
        :param root_logdir:
        :param no_train_metric:
        :param seed:
        :param debug:
        :return:
        """

        self.N = N
        self.T = T
        self.n_features = n_features
        self.hidden_layer_sizes = hidden_layer_sizes

        if not os.path.exists(root_savedir):
            os.makedirs(root_savedir)


        ###  Data handling  ###

        X_sp = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=[N, N])
        X_sp = X_sp + X_sp.transpose()
        X_sp = sp.triu(X_sp, k=1)
        row, col = X_sp.nonzero()

        pairs = get_pairs(N, row, col)
        pairs = pairs.astype(int)

        batch_generator = BatchGenerator(pairs, batch_size, holdout_ratio=holdout_ratio, seed=seed)


        ###  Construct the TF graph  ###

        self.construct_graph()

        print("Trainable variables:", tf.trainable_variables())

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-self.elbo)


        ###  Create q(Z) variational parameters  ###

        # before this was uniformly initialized
        # self.qZ_ = np.ones([N, T]) / T
        self.qZ_ = np.random.dirichlet(np.ones(T), size=N)  # (N, T)

        # the following quantity needs to be passed to the TF graph and must be updated after every update to qZ
        sum_qZ_above = np.zeros([N, T - 1])
        for k in range(T - 1):
            sum_qZ_above[:, k] = np.sum(self.qZ_[:, k + 1:], axis=1)


        ###  Training  ###

        if not no_train_metric:
            train_elbo = tf.placeholder(dtype=tf.float32, shape=[], name='train_elbo')
            train_elbo_summary = tf.summary.scalar('train_elbo', train_elbo)

            train_ll = tf.placeholder(dtype=tf.float32, shape=[], name='train_ll')
            train_ll_summary = tf.summary.scalar('train_ll', train_ll)

        if holdout_ratio is not None:
            test_ll = tf.placeholder(dtype=tf.float32, shape=[], name='test_ll')
            test_ll_summary = tf.summary.scalar('test_ll', test_ll)
        
        # Grab all scalar variables, to track in Tensorboard.
        trainable_vars = tf.trainable_variables()
        scalar_summaries = [tf.summary.scalar(tensor_.name, tensor_) for tensor_ in trainable_vars if len(tensor_.shape) == 0]
        tensor_summaries = [tf.summary.histogram(tensor_.name, tensor_) for tensor_ in trainable_vars if len(tensor_.shape) > 0]

        writer = tf.summary.FileWriter(root_logdir)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            init.run()

            if not no_train_metric:

                # add symmetric entries from the lower triangle
                train_data = batch_generator.train
                row = np.concatenate([train_data[:, 0], train_data[:, 1]])
                col = np.concatenate([train_data[:, 1], train_data[:, 0]])
                val = np.concatenate([train_data[:, 2], train_data[:, 2]])
                train_dict = {self.row: row, self.col: col, self.val: val, self.batch_scale: 1.0}

            if holdout_ratio is not None:
                test_data = batch_generator.test
                row = np.concatenate([test_data[:, 0], test_data[:, 1]])
                col = np.concatenate([test_data[:, 1], test_data[:, 0]])
                val = np.concatenate([test_data[:, 2], test_data[:, 2]])
                test_dict = {self.row: row, self.col: col, self.val: val, self.batch_scale: 1.0}

            for iteration in range(n_iterations):

                batch = batch_generator.next_batch()
                batch_dict = {self.row: np.concatenate([batch[:, 0], batch[:, 1]]),
                              self.col: np.concatenate([batch[:, 1], batch[:, 0]]),
                              self.val: np.concatenate([batch[:, 2], batch[:, 2]]),
                              self.qZ: self.qZ_,
                              self.n_samples: n_samples,
                              self.batch_scale: len(pairs) / len(batch),
                              self.sum_qZ_above: sum_qZ_above,
                              }

                # make a gradient update
                sess.run(train_op, feed_dict=batch_dict)

                # analytically
                self.update_qZ(sess=sess, batch=batch, n_samples=n_samples, debug=debug)


                # this update to sum_qZ_above was done at the beginning of the iteration. this implementation updates the sum_qZ_above before
                # logging the intermediate loss functions, and also one more time before saving the model. this actually makes more sense to me.
                # we could also just add this computation inside the construct graph function? it would have to be recomputed a few times more, but makes the code cleaner
                for k in range(T - 1):
                    sum_qZ_above[:, k] = np.sum(self.qZ_[:, k + 1:], axis=1)

                if iteration % 20 == 0:

                    print(iteration, end="")
                    
                    # Add scalar variables to Tensorboard.
                    for summ_str in sess.run(scalar_summaries):
                        writer.add_summary(summ_str, iteration)
                    # Add tensor variables to Tensorboard.
                    for summ_str in sess.run(tensor_summaries):
                        writer.add_summary(summ_str, iteration)

                    if not no_train_metric:
                        train_dict.update({self.qZ: self.qZ_, self.sum_qZ_above: sum_qZ_above, self.n_samples: 100})
                        train_ll_, train_elbo_ = sess.run([self.data_loglikel, self.elbo], feed_dict=train_dict)
                        train_ll_summary_str, train_elbo_summary_str = sess.run([train_ll_summary, train_elbo_summary],
                                                                                feed_dict={train_ll: train_ll_,
                                                                                           train_elbo: train_elbo_})
                        writer.add_summary(train_ll_summary_str, iteration)
                        writer.add_summary(train_elbo_summary_str, iteration)
                        print("\tTrain ELBO: %.4f" % train_elbo_, end="")
                        print("\tTrain LL: %.4f" % train_ll_, end="")

                    if holdout_ratio is not None:
                        test_dict.update({self.qZ: self.qZ_, self.sum_qZ_above: sum_qZ_above, self.n_samples: 100})
                        test_ll_ = sess.run(self.data_loglikel, feed_dict=test_dict)
                        test_ll_summary_str = sess.run(test_ll_summary, feed_dict={test_ll: test_ll_})
                        writer.add_summary(test_ll_summary_str, iteration)
                        print("\tTest LL: %.4f" % test_ll_)


            # save the model
            saver.save(sess, os.path.join(root_savedir, "model.ckpt"))

        # close the file writer
        writer.close()


    def update_qZ(self, sess, batch, n_samples, debug=False):

        """
        Analytically update the variational parameters of the distribution on the DP indicators Z.

        :param sess:
        :param qZ:
        :param batch:
        :param n_samples:
        :param debug:
        :return:
        """

        N, T = self.qZ_.shape

        # grab the values needed to update qZ
        E_log_V, E_log_1mV = sess.run([self.E_log_V, self.E_log_1mV])  # nothing needs to be passed to feed_dict

        # force symmetry in the subgraph so that updating the rows means updating all nodes in the subgraph
        row = np.concatenate([batch[:, 0], batch[:, 1]])
        col = np.concatenate([batch[:, 1], batch[:, 0]])
        val = np.concatenate([batch[:, 2], batch[:, 2]])

        # The terms corresponding to E[log p(X|Z,U)] are a bit tricky to compute; we update only those entries present in
        # the minibatch (i.e., the subgraph) and the likelihood is also approximated only on this minibatch.
        log_bernoulli_likel = sess.run(self.log_bernoulli_likel,
                                       feed_dict={self.row: row, self.col: col, self.val: val, self.n_samples: n_samples
                                                  })  # (n_samples, 2 * batch_size, n_T_pairs)

        all_T_pairs = np.concatenate([np.tril_indices(T, k=0),  # includes diagonal
                                      np.triu_indices(T, k=1)
                                      ], axis=1)  # (2, n_T_pairs)

        # this computation is an absolute nightmare... should think about if it's possible to change the way the TF graph
        # is constructed so that we can instead index (k, \ell) separately...
        qZ_col = self.qZ_[:, all_T_pairs[1]]  # (N, n_T_pairs)
        qZ_col = qZ_col[col, :]  # (2 * batch_size, n_T_pairs)
        ll_ = qZ_col * log_bernoulli_likel  # (n_samples, 2 * batch_size, n_T_pairs)

        ll_ = np.apply_along_axis(lambda x: np.bincount(row, weights=x, minlength=N),
                                  axis=1,
                                  arr=ll_)  # (n_samples, N, n_T_pairs)

        ll_ = np.apply_along_axis(lambda x: np.bincount(all_T_pairs[0], weights=x, minlength=T),
                                  axis=2,
                                  arr=ll_)  # (n_samples, N, T)

        ll_ = np.mean(ll_, axis=0)  # (N, T)

        if debug:

            # slow computation... that this computes the same thing is itself not that clear unfortunately...
            ll_test = np.zeros([N, T])

            for i in range(N):
                row_is_i = row == i  # (2 * batch_size,)

                if np.any(row_is_i):
                    log_p = log_bernoulli_likel[:, row_is_i, :]  # (n_samples, |E_i|, n_T_pairs); all terms correspond to when i is the row index
                    qZ_col = self.qZ_[col[row_is_i], :]  # (|E_i|, T); the q(Z_j=\ell) terms

                    for k in range(T):
                        row_is_k = all_T_pairs[0] == k
                        corresponding_col = all_T_pairs[1][row_is_k]
                        qZi_col_k = qZ_col[:, corresponding_col]  # (|E_i|, T) -- b/c k will be connected to T different \ell
                        ll_test[i, k] = np.sum(log_p[:, :, row_is_k] * qZi_col_k) / n_samples

            assert np.all(np.isclose(ll_, ll_test))

        # only replace for those nodes in the minibatch
        mbatch_row = np.unique(row)
        ll_ = ll_[mbatch_row, :]

        # the terms correspondin to E[log p(Z|V)] are a bit easier
        E_log_dp = np.zeros(T)
        for k in range(T - 1):
            E_log_dp[k] = E_log_V[k] + np.sum(E_log_1mV[:k])

        # final stick
        E_log_dp[-1] = np.sum(E_log_1mV)

        # should ll_ here be scaled by batch_scale?
        # ll_ = batch_scale * ll_ + E_log_dp
        ll_ = ll_ + E_log_dp  # (N, T)

        # now normalize to find the probability vectors
        Z_probs = np.exp(ll_ - logsumexp(ll_, axis=1)[:, None])

        # truncate anything too small
        to_truncate = Z_probs < 1e-8
        if np.any(to_truncate):
            Z_probs[to_truncate] = 1e-8
            Z_probs = Z_probs / np.sum(Z_probs, axis=1)[:, None]

        self.qZ_[mbatch_row, :] = Z_probs




if __name__=='__main__':

    N = 50
    X = np.random.rand(N,N)<0.4

    from scipy.sparse import find
    row, col, _ = find(X)

    root_savedir = "/Users/Koa/github-repos/bayes-nnet-mf/saved_sbm"
    root_logdir = os.path.join(root_savedir, 'tf_logs')

    if os.path.exists(root_logdir):
        shutil.rmtree(root_logdir)

    T = 10
    n_features = 8
    hidden_layer_sizes = [12, 8]


    m = SbmNNetMF()
    m.train(N, row, col, T=T, n_features=n_features, hidden_layer_sizes=hidden_layer_sizes, n_iterations=100, batch_size=50,
            n_samples=6, holdout_ratio=0.1, learning_rate=0.01, root_savedir=root_savedir, root_logdir=root_logdir,
            no_train_metric=False, seed=None, debug=False)

    os.system('~/anaconda3/bin/tensorboard --logdir=' + root_logdir)

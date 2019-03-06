import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
import gym.spaces

def _default_fn_create_optimizer(lr):
    return tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train, nsteps, ent_coef, vf_coef,
                max_grad_norm, fn_create_optimizer = _default_fn_create_optimizer):

        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + tf.losses.get_regularization_loss()
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = fn_create_optimizer(lr=LR, loss=loss)
        _train = trainer.apply_gradients(grads)

        def train(
                lr,
                cliprange,
                obs,
                returns,
                masks,
                actions,
                values,
                neglogpacs,
                states=None,
                timeout_in_ms=None
        ):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                if type(states) is dict:
                    for k, v in train_model.S.items():
                        td_map[v] = states[k]
                else:
                    td_map[train_model.S] = states
                td_map[train_model.M] = masks
            r = sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map,
                options=tf.RunOptions(timeout_in_ms=timeout_in_ms) if timeout_in_ms is not None else None
            )[:-1]
            return r
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        obs = env.reset()
        if type(env.observation_space) is gym.spaces.Tuple:
            self.obs = tuple(
                np.zeros(
                    (nenv,) + s.shape,
                    dtype=t.dtype.name
                ) for s, t in zip(env.observation_space.spaces, model.train_model.X)
            )
            for dst, src in zip(self.obs, obs):
                dst[:] = src
        else:
            shape = (nenv,) + env.observation_space.shape
            self.obs = np.zeros(shape, dtype=model.train_model.X.dtype.name)
            self.obs[:] = obs
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run_no_stats(self):
        for i in range(self.nsteps):
            actions, _, self.states, _ = self.model.step(self.obs, self.states, self.dones)
            obs, _, self.dones, _ = self.env.step(actions)
            if type(self.env.observation_space) is gym.spaces.Tuple:
                for self_obs_i, obs_i in zip(self.obs, obs):
                    self_obs_i[:] = obs_i
            else:
                self.obs[:] = obs

    def run(self):
        mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[]
        if type(self.env.observation_space) is gym.spaces.Tuple:
            mb_obs = tuple([] for _ in range(len(self.obs)))
        else:
            mb_obs = []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            if type(self.env.observation_space) is gym.spaces.Tuple:
                for i in range(len(self.obs)):
                    mb_obs[i].append(self.obs[i].copy())
            else:
                mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            obs, rewards, self.dones, infos = self.env.step(actions)
            if type(self.env.observation_space) is gym.spaces.Tuple:
                for dst, src in zip(self.obs, obs):
                    dst[:] = src
            else:
                self.obs[:] = obs
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        if type(self.env.observation_space) is gym.spaces.Tuple:
            mb_obs = tuple(np.asarray(i, dtype=o.dtype) for i, o in zip(mb_obs, self.obs))
        else:
            mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if type(arr) is tuple:
        r = tuple(i.swapaxes(0, 1).reshape(i.shape[0] * i.shape[1], *i.shape[2:]) for i in arr)
        return r
    else:
        s = arr.shape
        r = arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        return r

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0,

          total_timesteps=None,
          wall_t_end=None, wall_t_start=None,
          max_step_t=None,

          fn_create_optimizer=None,
          close_env = True,
          tf_timeout_in_ms = None
          ):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)

    assert total_timesteps is None or (type(total_timesteps) is int and total_timesteps > 0)
    assert (wall_t_end is None and wall_t_start is None) or \
        (type(wall_t_start) is float and type(wall_t_end) is float and wall_t_start >= 0. and wall_t_end > 0.)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    fn_create_optimizer = fn_create_optimizer)

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)

    if total_timesteps is not None:
        nupdates = total_timesteps//nbatch
    else:
        nupdates = None

    update = 0
    frac = 0.

    def save():
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i' % update)
        print('Saving to', savepath)
        model.save(savepath)

    assert nbatch % nminibatches == 0
    nbatch_train = nbatch // nminibatches

    def check_stability():
        if np.isnan(mblossvals).any():
            raise RuntimeError('NaN in PPO2 mblossvals ({} updates).'.format(update))
        if np.isinf(mblossvals).any():
            raise RuntimeError('Inf. in PPO2 mblossvals ({} updates).'.format(update))

    def terminate():
        check_stability()
        if save_interval:
            save()
        if close_env:
            env.close()
        return model

    runner_run_t_total = 0.
    t_first_update_start = time.time()
    while True:
        mblossvals = []

        t_update_start = time.time()

        update += 1

        if total_timesteps is not None and update == nupdates + 1:
            return terminate()

        if wall_t_end is not None:
            frac = min(1.,(t_update_start - wall_t_start) / (wall_t_end-wall_t_start))
        elif total_timesteps is not None:
            frac = max(frac, 1.0 - (update - 1.0) / nupdates)

        lrnow = lr(frac)

        cliprangenow = cliprange(frac)
        runner_run_t_start = time.time()
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        runner_run_t_end = time.time()
        if max_step_t != None and update > 1:
            runner_run_t_total += runner_run_t_end - runner_run_t_start
            avg_step_t = runner_run_t_total / (update * nbatch)
            if avg_step_t > max_step_t:
                raise RuntimeError('Runner avg_step_t of {} exceeded max_step_t of {}.'.format(avg_step_t, max_step_t))
            
        epinfobuf.extend(epinfos)
        if states is None: # nonrecurrent version
            # inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    # end = start + nbatch_train
                    # mbinds = inds[start:end]
                    slices_obs = tuple(arr[mbflatinds] for arr in obs)
                    slices = (arr[mbflatinds] for arr in (returns, masks, actions, values, neglogpacs))
                    mblossvals.append(
                        model.train(lrnow, cliprangenow, slices_obs, *slices, None, None, tf_timeout_in_ms)
                    )
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    if wall_t_end is not None and time.time() > wall_t_end:
                        return terminate()

                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices_obs = [arr[mbflatinds] for arr in obs]
                    slices = (arr[mbflatinds] for arr in (returns, masks, actions, values, neglogpacs))
                    if type(states) is dict:
                        mbstates = {k:v[mbenvinds] for k, v in states.items()}
                    else:
                        mbstates = states[mbenvinds]
                    mblossvals.append(
                        model.train(lrnow, cliprangenow, slices_obs, *slices, mbstates, tf_timeout_in_ms)
                    )

        check_stability()

        if len(mblossvals) > 0 and (update % log_interval == 0 or update == 1):
            lossvals = np.mean(mblossvals, axis=0)
            t_update_end = time.time()
            fps = int(nbatch / (t_update_end - t_update_start))
            ev = explained_variance(values, returns)

            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', t_update_end - t_first_update_start)
            logger.logkv('lr', lrnow)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            save()

    return terminate()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

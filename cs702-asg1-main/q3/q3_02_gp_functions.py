import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoNormal

from jax import random, grad

plt.style.use("seaborn-v0_8")

rng_key, rng_key_ = random.split(random.PRNGKey(0))

def get_data(simulation_function, rng_key, num_samples=100):
    
    rng_key, key_train, key_test = jax.random.split(rng_key, 3)
    
    X_train = jax.random.randint(key_train, (num_samples, 2), 0, 100)
    X_test = jax.random.randint(key_test, (num_samples, 2), 0, 100)
    
    Y_train = [simulation_function(x[0], x[1]) for x in X_train]
    Y_test  = [simulation_function(x[0], x[1]) for x in X_test]
    
    Y_train = jnp.array([[y] for y in Y_train])
    Y_test = jnp.array([[y] for y in Y_test])
    
    return X_train, Y_train, X_test, Y_test

def sort_vectors(X, Y, axis=0):
    # Get the indices that would sort X by the first column.
    sorted_indices = np.argsort(X[:, axis])
    # Use the sorted indices to reorder X and Y.
    X_sorted = X[sorted_indices]
    Y_sorted = Y[sorted_indices]
    return X_sorted, Y_sorted, sorted_indices

# squared exponential kernel with diagonal noise term
def exponential_kernel(X1, X2, var, length, noise, include_noise=True):
    
    # diff shape: (n_samples1, n_samples2, n_features)
    diff = X1[:, None, :] - X2[None, :, :]
    delta_sq = jnp.sum((diff / length) ** 2, axis=-1)
    
    k = var * jnp.exp(-0.5 * delta_sq)
    
    if include_noise:
        # Add a small constant to avoid exact zeros in the noise term
        noise = noise + 1.0e-6
        if X1.shape[0] == X2.shape[0]: # and jnp.allclose(X1, X2)
            k += noise * jnp.eye(X1.shape[0])
            
    return k

def expected_improvement(mu, s2, f_best):
    # Convert variance to standard deviation.
    sigma = jnp.sqrt(s2)
    
    # Improvement relative to the best observation.
    improvement = mu - f_best
    
    # Compute Z safely by handling the sigma==0 case.
    Z = jnp.where(sigma > 0, improvement / sigma, 0.0)
    
    # Standard expected improvement formula.
    ei = improvement * jax.scipy.stats.norm.cdf(Z) + sigma * jax.scipy.stats.norm.pdf(Z)
    
    # For sigma equal to zero, define EI as max(0, improvement).
    ei_mean = jnp.where(sigma == 0, jnp.maximum(0.0, improvement), ei)

    # Average EI across all observations.
    # ei_mean = jnp.mean(ei, axis=0)

    return ei_mean

def model(X, Y, kernel=exponential_kernel):
    
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # compute kernel
    k = kernel(X, X, var, length, noise)
    
    # sample Y according to the standard gaussian process formula
    numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k), obs=Y)

def predict(rng_key, X, Y, X_test, var, length, noise, kernel=exponential_kernel):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)

    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))

    var_pred = jnp.clip(jnp.diag(K), a_min=0.0)
    sigma = jnp.sqrt(var_pred)

    # compute mean
    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y)).squeeze()

    noise_sample = sigma * jax.random.normal(rng_key, shape=sigma.shape)
    sample = mean + noise_sample

    # Return the mean, the predictive variance (for EI), and a sample.
    return mean, var_pred, sample



def train_gp(rng_key, X, Y, NUM_ITERATIONS, LEARNING_RATE, NUM_CANDIDATE_SAMPLES, simulate_function):
    BEST_VALUE = jnp.max(Y)
    BEST_PARAMS = X[jnp.argmax(Y)]
    convergence_history = [BEST_VALUE]

    elbo_history = []
    var_history = []
    length_history = []
    noise_history = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n Iteration {iteration}")

        # --- 1. Model Fitting via SVI ---
        rng_key, rng_key_ = random.split(rng_key)
        guide = AutoNormal(model)
        svi = SVI(model, guide, optim.Adam(LEARNING_RATE), Trace_ELBO(), X=jnp.array(X), Y=jnp.array(Y))
        svi_result = svi.run(rng_key_, 1000)
        params = svi_result.params

        elbo_history.append(svi_result.losses[-1])

        # --- 2. Draw Posterior Samples ---
        rng_key, rng_key_ = random.split(rng_key)
        samples = guide.sample_posterior(rng_key_, params, sample_shape=(1000,))
        
        # Extract kernel hyperparameters from the posterior samples.
        rng_key, rng_key_ = random.split(rng_key)
        keys = random.split(rng_key_, samples["kernel_var"].shape[0])
        vars = samples["kernel_var"]
        lengths = samples["kernel_length"]
        noises = samples["kernel_noise"]
        
        # --- 3. Generate Candidate Points ---
        rng_key, rng_key_ = random.split(rng_key)
        X_candidates = random.randint(rng_key_, (NUM_CANDIDATE_SAMPLES, 2), 0, 101)

        # --- 4. Evaluate Predictive Means over Candidate Points ---
        means, variances, _ = jax.vmap(
            lambda rng_key, var, length, noise: predict(
                rng_key, X, Y, jnp.array(X_candidates), var, length, noise, kernel=exponential_kernel)
        )(keys, vars, lengths, noises)

        # --- 5. Compute Expected Improvement (EI) ---
        f_best = jnp.max(Y)
        # ei = expected_improvement(means.mean(axis=0), variances.mean(axis=0), f_best)
        ei_samples = jax.vmap(lambda mu, sigma: expected_improvement(mu, sigma, f_best))(means, variances)
        ei = ei_samples.mean(axis=0)  # Average over posterior samples

        # --- 6. Select the Next Point to Evaluate ---
        best_idx = jnp.argmax(ei)
        next_candidate = np.array(X_candidates)[best_idx]

        # --- 7. Evaluate the Objective Function using the Simulation ---
        new_value = simulate_function(next_candidate[0], next_candidate[1])
        
        print(f"Best observed value so far: {f_best}")
        print(f"Next candidate selected: {next_candidate}") # with EI: {ei[best_idx]}
        print(f"Simulation output at candidate: {new_value}")

        # --- 8. Update Training Data ---
        X = np.vstack([X, next_candidate])
        Y = np.append(Y, new_value)

        if new_value > BEST_VALUE:
            BEST_VALUE = new_value
            BEST_PARAMS = next_candidate

        convergence_history.append(new_value)
        var_history.append(jnp.mean(vars))
        length_history.append(jnp.mean(lengths))
        noise_history.append(jnp.mean(noises))

    # Documenting the convergence behavior:
    print("\n=== Convergence History (Best value over iterations) ===")
    for i, val in enumerate(convergence_history):
        print(f"Iteration {i}: {val:.4f}")

    return BEST_PARAMS, BEST_VALUE, guide, svi_result, elbo_history, var_history, length_history, noise_history, convergence_history
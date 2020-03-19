///  Gaussian Process Regression with Stan
///  Utilizes Cholesky Decomposition
///  Compound kernel
///  Allows for specification of a fixed prior mean
///  Heteroskedastic sigma
///  Hurdle Model at the lower bound
///  Vector prior mean
///  Multi company Model

functions {
  

  
  // Add a fixed constant to the add_diagonal for a square matrix
  matrix add_diagonal(matrix K, real delta){
    int N = rows(K);
    matrix[N,N] KK = K;
    
    for (i in 1:N){
      KK[i,i] = K[i,i] + delta;
    }
    
    return KK;
  } 


  // Square exponential kernel
  // no_input_vectors = 1 or 2: Use 1 if x1 and x2 are identical and are on the main diagonal (i.e.  for K, K**) 
  //                    otherwise use 2 (i.e. for K*)
  matrix cov_exp_quad_se(vector[] x1, vector[] x2, real eta_sq,  vector rho_sq, real delta, int no_input_vectors) {
    
    int N1 = size(x1);
    int N2 = size(x2);
    matrix[N1, N2] K;

    
    if( no_input_vectors == 1){
      for (i in 1:(N1-1)) {
        K[i, i] = eta_sq   + delta;
        for (j in (i+1):N2) {
          K[i, j] = eta_sq * exp(-0.5 * sum((x1[i] - x2[j]) .* ((x1[i] - x2[j])) ./ rho_sq));
          K[j, i] = K[i, j];
        }
      }
      K[N1, N1] = eta_sq   + delta;
    } else {
      for (i in 1:N1) {
        for (j in 1:N2) {
          K[i, j] = eta_sq * exp(-0.5 * sum((x1[i] - x2[j]) .* ((x1[i] - x2[j])) ./ rho_sq));
        }
      }    
    }
    return K;
  }
  


  // Calculates the categorical lkj factor kernel. 
  //
  // c1, c2 categorical values.  
  matrix cov_lkj(int[] c1, int[] c2, matrix lkj2){
    int n1 = size(c1);
    int n2 = size(c2);
    matrix[n1, n2] cat_mat ;
    
    for (i1 in 1:n1){
      for (i2 in 1:n2){
        cat_mat[i1,i2] = lkj2[c1[i1],c2[i2]];
      }
    }
    return cat_mat;
  }

  
  
  // General function to calculate an advance kernel
  // In this case, a square exponential kernel multiplied by a LKJ kernel
  matrix cov_fun(vector[] x1, vector[] x2, int[] x_cat1, int[] x_cat2,   
    real eta_sq, vector rho_sq,  matrix block_diag, real delta, int no_input_vectors){
      matrix[size(x1), size(x2)] res;

      res = cov_exp_quad_se(x1, x2, eta_sq,  rho_sq, delta,no_input_vectors);
      res = res .* block_diag;

    return res;
  }
  
  
  // prediction function
  vector gp_pred_rng(vector[] x1, vector[] x2, int[] x_cat1, int[] x_cat2, 
                     vector y1,  
                     real eta_sq, vector rho_sq,   matrix lkj2,
                     real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    real mu_vec1 = 0; // placeholder
    real mu_vec2 = 0; // placeholder
    
    vector[N2] f2;
    {
      matrix[N1, N1] K =   cov_fun(x1, x1, x_cat1, x_cat1, eta_sq, rho_sq, lkj2, delta, 1);
      matrix[N1, N1] L_K = cholesky_decompose(K);
      vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1-mu_vec1);
      vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
      matrix[N1, N2] k_x1_x2 = cov_fun(x1, x2, x_cat1, x_cat2, eta_sq, rho_sq, lkj2, 0, 2);
      vector[N2] f2_mu = mu_vec2 + (k_x1_x2' * K_div_y1);
      matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      matrix[N2, N2] cov_f2 = cov_fun(x2, x2, x_cat2, x_cat2, eta_sq, rho_sq, lkj2, delta,1) -v_pred' * v_pred;
      f2 = multi_normal_rng(f2_mu, cov_f2);
    }
    return f2;
  }
}

data {
  int<lower=1> N1;  // historic data
  int<lower=1> N2;  // predicted points
  int<lower=1> D; // Dimension of input values
  int<lower=1> dimCat; // dimension of number oif unique categorical values
  
  // Historic Data
  vector[D] x1[N1];  // input values (N1 x D matrix from R)
  int x_cat1[N1];  // input values containing a categorical value
  
  vector[N1] y1;  // output values 

  // Inputs for prediction
  vector[D] x2[N2];  // input values (N x D matrix from R)
  int x_cat2[N2];  // input values containing a categorical value

 // inputs for priors on parameters
 real<lower=0> prior_eta;
 real prior_rho[2];
 real<lower=0> prior_sigma;
 real<lower=0> prior_alpha;
}

transformed data{
  real delta = 0.000001;
  matrix[N1, N1] block_diag = rep_matrix(0, N1,N1);
  
  for (i in 1:N1){
    for (j in 1:N1){
      if (x_cat1[i] == x_cat1[j]) block_diag[i,j] =1;
    }
  }
}  

parameters {
  vector<lower=0>[D] rho;
  real<lower=0> eta;
  vector<lower=0>[dimCat] sigma;

  vector[N1] z;
  
  
}

transformed parameters {
  vector<lower=0>[D] rho_sq;
  real<lower=0> eta_sq;

  eta_sq = eta ^2;
  for (i in 1:D){
    rho_sq[i] = rho[i]^2;
  }
  
}

model {
  
  vector[N1] f;
  vector[N1] sigma_long;
  {
  
    // Mean field function
    matrix[N1, N1] cov =  cov_fun(x1, x1, x_cat1, x_cat1,  eta_sq, rho_sq, block_diag, delta, 1);
    matrix[N1, N1] L_cov;
    
    for (i in 1:N1){
      sigma_long[i] =  sigma[x_cat1[i]];
    }
    
    L_cov = cholesky_decompose(cov);
    
    
    f = L_cov * z;
  }
  

  // Priors
  rho ~ inv_gamma(prior_rho[1], prior_rho[2]);
  eta ~ normal(0, prior_eta);
  sigma ~ normal(0, prior_sigma);
  z ~ std_normal();


  // likelihood for normal distribution truncated to left at hurdle
  
  y1 ~ normal(f, sigma_long);
  
}

generated quantities {
  vector[N2] fStar = gp_pred_rng(x1, x2, x_cat1, x_cat2, y1,
                                  eta_sq, rho_sq, block_diag, delta);
  // vector[N2] yStar;
  // for (n in 1:N2)
  //   yStar[n] = hurdle_rng(fStar[n], sigma[dimSigma+2-dev_lag2[n]], l_bound);
}

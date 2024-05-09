"""
This is the main script of the simulation
"""

# Set path to data
  include("data_path.jl")

# Simulation parameter
  ncores   = 3           # Number of cores (for workers) 	
  N    	   = Int64(10)  # Number of Monte Carlo iterations 
  dataset  = 0           # 0 = Financial data, 1 = Macroeconomic data (no lags), 2 = Macroeconomic data (including 4 lags)
  err_type = 1           # 0 = normal errors,  1 = t-distributed errors 
  diag_cov = false       # Use diagonal covariance matrix?
  q0       = Int64(140)  
  τ0       = Int64(60)

# Set parameters for GLP code (N_glp = burnin sample)
  N_glp = Int64(1e3)	
  M_glp	= Int64(10e3) + N_glp

# Run script to load packages and prepare data (run only once (!))
  include("PrepareData.jl")

# Include scripts with functions
  include("GLP_SpikeSlab.jl")
  include("Functions.jl")

# Run functions once for compilation
  include("Compile_functions.jl")

# Make chunks for tasks
N_chunks = Iterators.partition(1:N, div(N, nworkers())) |> collect


#-----------------------------------------------------------------------------------------------------#	
#---------------------------------------- Begin Simulation -------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
for jj = 1:length(ω)
    for ii = 1:length(nz_β)

# Glmnet results               
  ridge_mse[ii, jj] = @views 1 - mean(first.(results_glmnet[:, 1]))/MSE_hmean
  enet_mse[ii, jj]  = @views 1 - mean(first.(results_glmnet[:, 2]))/MSE_hmean
  lasso_mse[ii, jj] = @views 1 - mean(first.(results_glmnet[:, 3]))/MSE_hmean

# Avg. number of included predictors 
  enet_nr[ii,  jj]   = @views mean(getindex.(results_glmnet[:, 2], 3))
  lasso_nr[ii, jj]	 = @views mean(getindex.(results_glmnet[:, 3], 3))

# Get true positives                 
  enet_tp[ii, jj]  = mean(getindex.(results_glmnet[:, 2], 4))
  lasso_tp[ii, jj] = mean(getindex.(results_glmnet[:, 3], 4))

# Relaxed Lasso
  lasso_relax_mse[ii, jj]  = @views 1 - mean(getindex.(results_relax, 1))/MSE_hmean
  lasso_relax_nr[ii, jj]   = @views mean(getindex.(results_relax, 3))
  lasso_relax_tp[ii, jj]   = mean(getindex.(results_relax, 5))

# Adaptive Lasso
  lasso_adapt_mse[ii, jj] = @views 1 - mean(getindex.(results_adlasso, 1))/MSE_hmean
  lasso_adapt_nr[ii, jj]  = @views mean(getindex.(results_adlasso, 3))
  lasso_adapt_tp[ii, jj]  = mean(getindex.(results_adlasso, 4))

# GLP results
  glp_mse[ii, jj]        = 1 - @views mean(first.(results_glp_all))/MSE_hmean
  glp_nr[ii, jj]         = @views round(mean(getindex.(results_glp_all, 3)), digits = 4)
  glp_tp[ii, jj]         = mean(getindex.(results_glp_all, 4))

# BSS results 
  bss_mse[ii, jj]        = 1 - mean(getindex.(bkm_results_all, 1))/MSE_hmean		   
  bss_nr[ii, jj]         = mean(getindex.(bkm_results_all, 2))
  bss_mse_shrunk[ii, jj] = 1 - mean(getindex.(bkm_results_all, 3))/MSE_hmean		
  bss_tp[ii, jj]         = mean(getindex.(bkm_results_all, 4))


  end
end

pmap(i -> simul_func(βx, ω, Σ_x, ϕx, μx, nz_β, n, ii, jj, i), N_chunks)

include("test.jl")
@everywhere include("test.jl")
@everywhere begin
    βx = $βx
    ω  = $ω 
    Σ_x = $Σ_x
    ϕx  = $ϕx
    μx = $μx
    nz_β = $nz_β
    n = $n
    ii = $ii
    jj = $jj
end

simul_func(βx, ω, Σ_x, ϕx, μx, nz_β, n, ii, jj, 1:5)

function simul_func(βx, ω, Σ_x, ϕx, μx, nz_β, n, ii, jj, N_chunk)
#------------------------------------- Simulate data -----------------------------------------------#
  # Simulate data
    β_active    = map(kk -> sample(1:length(βx), nz_β[ii]; replace = false), N_chunk) # Draw active predictors	  
    xysim_data  = map(kk -> data_simul(n, βx , Σ_x, μx, ω[jj], β_active[kk], ϕx, 
                                       err_type, ν, (kk + ii)), N_chunk)                          																 

  # Compute MSE for historical mean	
    MSE_hmean = @views mean(map(kk -> (mean(xysim_data[kk][1:(end - 1), 1]) - 
                                            xysim_data[kk][end, 1])^2, N_chunk))

  # Merge simulated data and active predictors                                          
    xydata_βact     = [xysim_data β_active]                                             

#------------------------------------- Glmnet  ----------------------------------------------#	
# Make all combinations of α and N
  αN_comb = collect(Iterators.product(N_chunk, α));	

# Results for weak predictors
  results_glmnet = pmap(αN_comb) do kk

                  glmnet_simul_cvts(xysim_data[kk[1]][:, 1], 
                                    xysim_data[kk[1]][:, 2:end], 
                                    q0, 
                                    τ0, 
                                    kk[2],
                                    β_active[kk[1]])                                  
                                                              
              end        

#------------------------------------- Relax Lasso  ----------------------------------------------#	 

# Results for relaxed Lasso                                                      
  results_relax = pmap(eachrow(xydata_βact)) do kk

                glmnet_relaxed_cvts(kk[1][:, 1], 
                                    kk[1][:, 2:end], 
                                    q0, τ0, 1.0, ζ,
                                    kk[2])

                  end
 
#------------------------------------- Adaptive Lasso  ----------------------------------------------#	
 # Results for relaxed Lasso                                                      
   results_adlasso = pmap(eachrow(xydata_βact)) do kk

              adaptive_lasso_cvts(kk[1][:, 1], 
                                  kk[1][:, 2:end], 
                                  q0, τ0,
                                  kk[2])

        end

#--------------------------------------------- GLP -----------------------------------------------#

# GLP results
  results_glp_all  = pmap(eachrow(xydata_βact)) do kk 

                      GLP_oos(kk[1][1:(end - 1), 2:end], kk[1][end, 2:end], 
                              kk[1][1:(end - 1), 1], kk[1][end, 1],
                              M_glp, N_glp, abeta, bbeta, Abeta, Bbeta, 
                              Int64(floor(abs(sum(kk[1])))),
                              kk[2])

                      end
   
#-------------------------------------------- BSS  ---------------------------------------------#
  # Train and test observations
    train_seq = @views collect(1:(size(xysim_data[1], 1) - 1))
    obs_test  = @views size(xysim_data[1], 1)[1]

  # Compute models
    bkm_results_all  = pmap(kk -> bkm_predict(kk[1], train_seq, obs_test, kk[2]), eachrow(xydata_βact)) 

  # Show progress
    @info string("ω = ", ω[jj], "; ", "nz_β = ", nz_β[ii])

  # Return named tuple  
    return (results_glmnet = results_glmnet, 
            results_relax  = results_relax, 
            results_adlasso = results_adlasso, 
            results_glp_all = results_glp_all, 
            bkm_results_all = bkm_results_all, 
            MSE_hmean = MSE_hmean)
   
end    

#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

# Pre-allocate matrices to store results 
  mat_mse  	       = zeros(1, length(ω) + 1)
  mat_shrunk_mse	 = zeros(1, length(ω) + 1)
  mat_nr           = zeros(1, length(ω) + 1)
  mat_tp           = zeros(1, length(ω) + 1)
  zero_vec         = zeros(1, length(ω)+ 1)

# Add row names (MSE unshrunk) 
  fccomb_flex_mse  = hcat(repeat([string("FC\$^{\\text{Flex}}\$")], length(nz_β)), fccomb_flex_mse)
  fccomb_ew_mse    = hcat(repeat([string("FC\$^{\\text{EW}}\$")], length(nz_β)), fccomb_ew_mse)
  bss_mse          = hcat(repeat([string("Best Subset")],       length(nz_β)), bss_mse)
  lasso_mse        = hcat(repeat([string("Lasso")],             length(nz_β)), lasso_mse)
  lasso_relax_mse  = hcat(repeat([string("Relaxed Lasso")],     length(nz_β)), lasso_relax_mse)
  lasso_adapt_mse  = hcat(repeat([string("Adaptive Lasso")],    length(nz_β)), lasso_adapt_mse)
  enet_mse         = hcat(repeat([string("Elastic Net")],       length(nz_β)), enet_mse)
  glp_mse          = hcat(repeat([string("Bayesian FSS")],      length(nz_β)), glp_mse)
  ridge_mse        = hcat(repeat([string("Ridge")],             length(nz_β)), ridge_mse)

# Add row names (MSE shrunk) 
  fccomb_flex_mse_shrunk = hcat(fccomb_flex_mse[:, 1],  fccomb_flex_mse_shrunk)
  fccomb_ew_mse_shrunk   = hcat(fccomb_ew_mse[:, 1],    fccomb_ew_mse_shrunk)
  bss_mse_shrunk         = hcat(bss_mse[:, 1],          bss_mse_shrunk)
  lasso_mse_shrunk       = hcat(lasso_mse[:, 1],        lasso_mse_shrunk) 
  lasso_relax_mse_shrunk = hcat(lasso_relax_mse[:, 1],  lasso_relax_mse_shrunk)
  lasso_adapt_mse_shrunk = hcat(lasso_adapt_mse[:, 1],  lasso_adapt_mse_shrunk)
  enet_mse_shrunk        = hcat(enet_mse[:, 1],         enet_mse_shrunk)
  glp_mse_shrunk         = hcat(glp_mse[:, 1],          glp_mse_shrunk)                            
  ridge_mse_shrunk       = hcat(ridge_mse[:, 1],        ridge_mse_shrunk)			
  
# Add row names (Average number of predictors) 
  fccomb_flex_nr = hcat(fccomb_flex_mse[:, 1],  fccomb_flex_nr)
  bss_nr         = hcat(bss_mse[:, 1],          bss_nr)
  lasso_relax_nr = hcat(lasso_relax_mse[:, 1],  lasso_relax_nr)
  lasso_adapt_nr = hcat(lasso_adapt_mse[:, 1],  lasso_adapt_nr)
  lasso_nr       = hcat(lasso_mse[:, 1],        lasso_nr)    
  enet_nr        = hcat(enet_mse[:, 1],         enet_nr)
  glp_nr         = hcat(glp_mse[:, 1],          glp_nr)                            
  	
# Add row names (True positives) 
  fccomb_tp        = hcat(fccomb_flex_mse[:, 1], fccomb_tp)
  bss_tp           = hcat(bss_mse[:, 1],         bss_tp)
  lasso_relax_tp   = hcat(lasso_relax_mse[:, 1], lasso_relax_tp)
  lasso_adapt_tp   = hcat(lasso_adapt_mse[:, 1], lasso_adapt_tp)
  lasso_tp         = hcat(lasso_mse[:, 1],       lasso_tp)    
  enet_tp          = hcat(enet_mse[:, 1],        enet_tp)
  glp_tp           = hcat(glp_mse[:, 1],         glp_tp)           
  

 
# Loop to fill result matrices  
  for ii = 1:length(nz_β)

    # Fill matrices with MSE
    global mat_mse = vcat(mat_mse,
                          reshape(fccomb_flex_mse[ii,  :], 1, :),
                          reshape(fccomb_ew_mse[ii,  :], 1, :),
                          reshape(bss_mse[ii,  :], 1, :),
                          reshape(lasso_relax_mse[ii,  :], 1, :),
                          reshape(lasso_adapt_mse[ii,  :], 1, :),
                          reshape(lasso_mse[ii,  :], 1, :),
                          reshape(enet_mse[ii,  :], 1, :),
                          reshape(glp_mse[ii,  :], 1, :),
                          reshape(ridge_mse[ii,  :], 1, :),
                          zero_vec,															
                          ) 

    global mat_shrunk_mse = vcat(mat_shrunk_mse,
                                reshape(fccomb_flex_mse_shrunk[ii,  :], 1, :),
                                reshape(fccomb_ew_mse_shrunk[ii,  :], 1, :),
                                reshape(bss_mse_shrunk[ii,  :], 1, :),
                                reshape(lasso_relax_mse_shrunk[ii,  :], 1, :),
                                reshape(lasso_adapt_mse_shrunk[ii,  :], 1, :),
                                reshape(lasso_mse_shrunk[ii,  :], 1, :),
                                reshape(enet_mse_shrunk[ii,  :], 1, :),
                                reshape(glp_mse_shrunk[ii,  :], 1, :),
                                reshape(ridge_mse_shrunk[ii,  :], 1, :),                              													
                                 zero_vec) 
  
  # Fill matrices with number of chosen predictors 									
    global mat_nr = vcat(mat_nr,
                         reshape(fccomb_flex_nr[ii,  :], 1, :),                    
                         reshape(bss_nr[ii,  :], 1, :),
                         reshape(lasso_relax_nr[ii,  :], 1, :),
                         reshape(lasso_adapt_nr[ii,  :], 1, :),
                         reshape(lasso_nr[ii,  :], 1, :),
                         reshape(enet_nr[ii,  :], 1, :),
                         reshape(glp_nr[ii,  :], 1, :),  																	
                         zero_vec) 

    global mat_tp = vcat(mat_tp,
                         reshape(fccomb_tp[ii,  :], 1, :),                    
                         reshape(bss_tp[ii,  :], 1, :),
                         reshape(lasso_relax_tp[ii,  :], 1, :),
                         reshape(lasso_adapt_tp[ii,  :], 1, :),
                         reshape(lasso_tp[ii,  :], 1, :),
                         reshape(enet_tp[ii,  :], 1, :),
                         reshape(glp_tp[ii,  :], 1, :),  			                  											
                         zero_vec) 

  end

# Round slice matrices
  mat_mse        = @views mat_mse[2:end, :]
  mat_shrunk_mse = @views mat_shrunk_mse[2:end, :]
  mat_nr         = @views mat_nr[2:end, :]
  mat_tp         = @views mat_tp[2:end, :]

# Merge matrices 
  mat_merge_mse  = @views vcat(mat_mse, mat_shrunk_mse)  
  mat_merge_nrtp = @views vcat(mat_nr, mat_tp)  
 
# Convert matrices to R objects	
  omega = ω
  @rput omega
  @rput mat_merge_mse   
  @rput mat_merge_nrtp         
  @rput err_type

# Convert Matrix to latex table (with R)
  R"""
  library(xtable)

  # MSE-Matrix for weak predictors
    table_mse					  <- mat_merge_mse
    colnames(table_mse) <- c("", paste("omega = ", omega)) 
    table_mse           <- xtable(table_mse, digits = 4)
     
    
  # Number of variables for weak predictors
    table_nr 				   <- mat_merge_nrtp
    colnames(table_mse) <- c("", paste("omega = ", omega)) 
    table_nr           <- xtable(table_nr, digits = 4)
  
  # Save all tables in list
    all_tables = list((table_mse), 
                      (table_nr))
    print(all_tables, sanitize.text.function = function(x) x, 
                      include.rownames = FALSE, 
                      booktabs = TRUE)                      

    dstring <- if(err_type == 0) "_DN.RData" else "_DT.RData"
    save(all_tables, file = paste("SimTables_finance_18_03_23", dstring, sep = ""))
  """

 # Close workers
   rmprocs(workers())




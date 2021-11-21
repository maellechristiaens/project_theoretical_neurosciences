
# ------------------------------------------------------------------------
# def transform_to_array(v, T, nbx, nbs=nbs):
#     dim_v = len(v)
#     if dim_v == nbs:
#         V1 = np.stack([v for _ in range(nbx)], axis=0) # dim0: x, dim1: k
#         V2 = np.stack([V1 for _ in range(T)], axis=2) # dim0: x, dim1: k, dim2: T
#     elif dim_v == nbx:
#         V1 = np.stack([v for _ in range(nbs)], axis=1) # dim0: x, dim1: k
#         V2 = np.stack([V1 for _ in range(T)], axis=2) # dim0: x, dim1: k, dim2: T
#     elif dim_v == T:
#         V0 = np.stack([v for _ in range(nbs)], axis=0) # dim0: k, dim1: T
#         V1 = np.stack([V0 for _ in range(nbx)], axis=2) # dim0: k, dim1: T, dim2: X
#         V2 = np.transpose(V1, axes=(2,0,1)) # dim0: k, dim1: T, dim2: X
#     else:
#         print('Error')
#     return V2

# def ornstein_uhlenbeck_process(X, cR, cL, lbda=lbda, dt=dt, dx=dx, nbs=nbs, renormalize=True, check=True):
#     T = len(cR)
#     nbx = len(X) # number of accumulator values, including 2 extra bins outside boundaries
    
#     k = np.arange(-int(nbs/2), int(nbs/2)+1)
#     c_in = total_input(cR, cL)
#     sgmdt_in = total_variance(cR, cL, sgm_a, sgm_s, dt)

#     K = transform_to_array(k, T, nbx, nbs)
#     X_exp = transform_to_array(X, T, nbx, nbs)
#     C = transform_to_array(c_in, T, nbx, nbs)
#     SGM = transform_to_array(sgmdt_in, T, nbx, nbs)

#     M = deterministic_drift(X_exp, C, lbda, dt)
#     S0 = positions(SGM, K, nbs)
#     S = M + S0
#     Ps = gaussian(K, nbs)
#     if renormalize :
#         norm = np.sum(Ps[0,:,0])
#         Ps /= norm
#     Ps[0,1:,:] = 0
#     Ps[0,0,:] = 1
#     Ps[-1,:-1,:] = 0
#     Ps[-1,-1,:] = 1

#     if check :
#         print('Sum Ps : ', np.sum(Ps[:,:,0], axis=1))
#         # plt.plot(C[2,0,:])
#         # plt.show()
#     return S, Ps

# def forward_transition_matrix0(X, S, Ps, check=False):
#     nbx = len(X)
#     T = S.shape[2]
#     F = np.zeros((nbx,nbx,T))
#     # Transient states
#     for i in range(1,nbx-1): 
#         print(i)
#         mask_low = (S>X[i-1]) & (S<X[i])
#         mask_up = (S>X[i]) & (S<X[i+1])
#         mask_eq = (S==X[i])
#         S_low = S*mask_low
#         S_up = S*mask_up
#         S_eq = S*mask_eq
#         Ps_low = Ps*mask_low
#         Ps_up = Ps*mask_up
#         Ps_eq = Ps*mask_eq
#         W_low = split_mass_low(S_up, X[i], X[i+1])
#         W_up = split_mass_up(S_low, X[i-1], X[i])
#         P_tot = np.sum(W_up*Ps_low, axis=1) + np.sum(W_low*Ps_up, axis=1) + np.sum(Ps_eq, axis=1)
#         F[i,:,:] = P_tot # P(a(t)=xi | a(t-1)=xj) -> no need to sum to 1
#     # Absorbant state i = 0
#     mask_low = (S<=X[0])
#     mask_up = (S>X[0]) & (S<X[1])
#     S_low = S*mask_low
#     S_up = S*mask_up
#     Ps_low = Ps*mask_low
#     Ps_up = Ps*mask_up
#     W_low = split_mass_low(S_up, X[0], X[1])
#     P_tot = np.sum(Ps_low, axis=1) + np.sum(W_low*Ps_up, axis=1)
#     F[0,:,:] = P_tot
#     F[0,0,:] = 1 # absorbant state
#     # Absorbant state i = nbx-1
#     mask_low = (S>X[-2]) & (S<X[-1])
#     mask_up = (S>=X[-1])
#     S_low = S*mask_low
#     S_up = S*mask_up
#     Ps_low = Ps*mask_low
#     Ps_up = Ps*mask_up
#     W_up = split_mass_up(S_low, X[-2], X[-1])
#     P_tot = np.sum(W_up*Ps_low, axis=1) + np.sum(Ps_up, axis=1)
#     F[-1,:,:] = P_tot
#     F[-1,-1,:] = 1
#     if check :
#         print(np.sum(F[:,:,0], axis=0)) # sum across all lines for each column
#         # i.e. sum over all landing positions ai for each initial position aj
#     return F


# ------------------------------------------------------------------------

# def transform_array_4D(V):
#     V2 = np.stack([V for _ in range(V.shape[0])], axis=3)
#     return V2

# def forward_transition_matrix2(X, S, Ps, check=False):
#     nbx = len(X)
#     T = S.shape[2]
#     F = np.zeros((nbx,nbx,T))
#     # Transient states
#     print(X.shape)
#     X_c = transform_array_4D(X[1:-2,:]) # dim0: xi, dim1: k, dim2: T, dim3: xj
#     X_up = transform_array_4D(X[2:-1,:])
#     X_low = transform_array_4D(X[0:-2,:])
#     S_c = transform_array_4D(S[1:-2,:])
#     P_tot = transform_array_4D(Ps[1:-2,:])
#     mask_low = (S_c>X_low) & (S_c<X_c)
#     mask_up = (S_c>X_c) & (S_c<X_up)
#     mask_eq = (S_c==X_c)
#     # S_low = S_c*mask_low
#     # S_up = S_c*mask_up
#     # S_eq = S_c*mask_eq
#     Ps_low = Ps*mask_low
#     P_s_up = Ps*mask_up
#     P_s_eq = Ps*mask_eq
#     W_low = split_mass_low(S_c, X_c, X_up)
#     W_up = split_mass_up(S_c, X_low, X_c)
#     F[1:-2,:,:] = np.sum(W_up*Ps_low, axis=2) + np.sum(W_low*Ps_up, axis=2) + np.sum(Ps_eq, axis=2) # sum over dim2: k

# ------------------------------------------------------------------------

# def split_mass(s, x_low, x_up, side="up"):
#     norm = x_up - x_low +1
#     p_low = (x_up - s)/norm
#     p_up = (s - x_low)/norm
#     if 'up':
#         return p_up
#     else:
#         return p_low

# def forward_transition_matrix3(X, S, Ps, check=False):
#     nbx = len(X)
#     T = S.shape[2]
#     F = np.zeros((nbx,nbx,T))
#     # Transient states
#     P0 = Ps[:,:,0]
#     i_low = np.zeros(S.shape)
#     X_low = np.zeros(S.shape)
#     X_up = np.zeros(S.shape)
#     X_eq = np.zeros(S.shape)
#     for i in range(1,nbx-1):
#         # print(i)
#         mask = (S>=X[i])&(S<X[i+1])
#         i_low[mask] = i
#         X_low[mask] = X[i]
#         X_up[mask] = X[i+1]
#     W_low = split_mass(S, X_low, X_up, 'low')
#     W_up = split_mass(S, X_low, X_up, 'up')
#     # print(i_low)
#     # print(W_low)
#     for i in range(len(X)):
#         print(i)
#         for j in range(len(X)):
#             for t in range(T):
#                 mask_low = (i_low[j,:,t]==i)
#                 mask_up = (i_low[j,:,t]==i-1)
#                 # print(mask_low.shape)
#                 # print(W_low[j,:,t].shape)
#                 # print(P0[j].shape)
#                 # print((P0[j]*W_low[j,:,t]).shape)
#                 # print((P0[j]*W_low[j,:,t])[mask_low])
#                 # print(np.sum((P0[j]*W_low[j,:,t])[mask_low]))
#                 F[i,j,t] = np.sum((P0[j]*W_low[j,:,t])[mask_low])
#     return F


# ------------------------------------------------------------------------


# def forward_transition_matrix4(X, S, Ps, check=False):
#     nbx = len(X)
#     T = S.shape[2]
#     F = np.zeros((nbx,nbx,T))
#     # Transient states
#     P0 = Ps[:,:,0]
#     # for i in range(1,nbx-1):
#         print(i)
#         i_j, i_s, i_t = np.where((S>=X[i])&(S<X[i+1]))
#         print(i_j, i_s, i_t)
#         print(S[i_j, i_s, :].shape)
#         for j in i_j:
#             for t in i_t:
#                 W_low = split_mass(S[j, i_s, t], X[i], X[i+1], 'low')
#                 W_up = split_mass(S[j, i_s, t], X[i], X[i+1], 'up')
#                 F[i,j,t] = np.sum((P0[j, i_s]*W_low))
#     return F
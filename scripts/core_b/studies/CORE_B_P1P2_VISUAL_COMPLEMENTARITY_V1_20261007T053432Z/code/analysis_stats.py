from __future__ import annotations
import numpy as np

def shared_group_bootstrap_counts(group_count:int,replicates:int,seed:int)->np.ndarray:
 if group_count<1 or replicates<1:raise ValueError('positive group and replicate counts required')
 rng=np.random.default_rng(seed)
 draws=rng.integers(0,group_count,size=(replicates,group_count),dtype=np.int32)
 counts=np.zeros((replicates,group_count),dtype=np.uint16)
 rows=np.broadcast_to(np.arange(replicates,dtype=np.int32)[:,None],draws.shape)
 np.add.at(counts,(rows,draws),1)
 return counts

def episode_equal_cluster_means(group_sums:np.ndarray,group_episode_counts:np.ndarray,draw_counts:np.ndarray)->np.ndarray:
 sums=np.asarray(group_sums,dtype=np.float64);episodes=np.asarray(group_episode_counts,dtype=np.float64);counts=np.asarray(draw_counts,dtype=np.float64)
 if sums.ndim!=1 or episodes.shape!=sums.shape or counts.ndim!=2 or counts.shape[1]!=len(sums):raise ValueError('cluster bootstrap dimensions mismatch')
 denom=counts@episodes
 if np.any(denom<=0):raise ValueError('bootstrap draw contains no episodes')
 return (counts@sums)/denom

def paired_cluster_bootstrap_delta(model_group_sums:np.ndarray,control_group_sums:np.ndarray,group_episode_counts:np.ndarray,draw_counts:np.ndarray)->np.ndarray:
 m=episode_equal_cluster_means(model_group_sums,group_episode_counts,draw_counts)
 c=episode_equal_cluster_means(control_group_sums,group_episode_counts,draw_counts)
 return m-c

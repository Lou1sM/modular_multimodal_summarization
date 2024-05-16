import numpy as np
from episode import episode_from_epname


def names_in_scene(scene):
    return set([line.split(':')[0] for line in scene.strip('\n').split('\n')
                if not line.startswith('[')])

def identical_char_names(scene1, scene2):
    cn1 = [x for x in names_in_scene(scene1) if x!='']
    cn2 = [x for x in names_in_scene(scene2) if x!='']
    return '$'.join(sorted(cn1)) == '$'.join(sorted(cn2))

def optimal_order(scenes):
    char_names_in_scenes = [names_in_scene(scene) for scene in scenes]
    n = len(scenes)
    dists = np.empty([n,n])
    #print('\n'.join([', '.join(cs) for cs in char_names_in_scenes]))
    for i, cn1 in enumerate(char_names_in_scenes):
        dists[i,i] = 0
        for j in range(i+1,n):
            cn2 = char_names_in_scenes[j]
            n_overlap = sum([x in cn2 for x in cn1])
            iou = 0 if (len(cn1)+len(cn2)==0) else n_overlap / (len(cn1)+len(cn2))
            dists[i,j] = 1 - iou
            dists[j,i] = 1 if n_overlap==0 else np.inf # never change order for a char
    dists = np.concatenate([dists,np.zeros([n,1])],axis=1) #dummy endpoint

    order_idxs = list(range(n+1)) #dummy endpoint
    while True:
        changed = False
        for i in range(2,n):
            prevs_with_matching_chars = [j for j,scene_idx in enumerate(order_idxs[:i]) if dists[scene_idx,order_idxs[i]]<1]
            if len(prevs_with_matching_chars) > 0: # move as far to left without inverting char order
                new_pos = max(prevs_with_matching_chars)+1
                if new_pos==i:
                    continue
                cost_improvement_at_i = dists[order_idxs[i-1],order_idxs[i]] + \
                                        dists[order_idxs[i],order_idxs[i+1]] - \
                                        dists[order_idxs[i-1],order_idxs[i+1]]

                cost_improvement_at_new_pos = dists[order_idxs[new_pos-1],order_idxs[new_pos]] - \
                                              dists[order_idxs[new_pos-1],order_idxs[i]] - \
                                              dists[order_idxs[i],order_idxs[new_pos]]

                if cost_improvement_at_i + cost_improvement_at_new_pos <= 0:
                    continue
                old_costs = [dists[order_idxs[k],order_idxs[k+1]] for k in range(n-1)]
                to_move = order_idxs.pop(i)
                order_idxs.insert(new_pos,to_move)
                changed = True
                new_costs = [dists[order_idxs[k],order_idxs[k+1]] for k in range(n-1)]
                if not np.allclose(sum(old_costs)-sum(new_costs), cost_improvement_at_i+cost_improvement_at_new_pos):
                    breakpoint()
        if not changed:
            break
        #print('\n'.join([', '.join(char_names_in_scenes[oi]) for oi in order_idxs[:-1]]))
    return order_idxs

if __name__ == '__main__':
    #ep = episode_from_epname('bb-01-10-14')
    ep = episode_from_epname('atwt-01-02-03')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    optimal_order(ep.scenes)

import matplotlib.pyplot as plt
import pickle


def smoothing(lst):


    ret=[]

    for i in range(len(lst)):
        upper=min(i+25,1999)
        lower=max(i-25,0)
        _sum=0
        for v in range(lower,upper):
            _sum=_sum+lst[v]

        average=_sum/(upper-lower)
        ret.append(average)
    return ret

# with open('q_diff.pickle', 'rb') as f:
#     ep = pickle.load(f)
with open('q_diff_alpha.pickle', 'rb') as f:
    ep = pickle.load(f)


key=ep.keys()
print(key)
for epsilone in key:
    plt.plot(smoothing(ep[epsilone]))

plt.legend(key)
# plt.title('Q-diff with various epsilone')
plt.title('Q-diff with various alpha')
plt.xlabel('episode')
plt.ylabel('Q-value difference')
plt.axhline(y=0, color='r', linewidth=1)

# plt.savefig('fig.png')
plt.savefig('fig_alpha.png')




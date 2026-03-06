

{0}------------------------------------------------

# Discounted Thompson Sampling for Non-Stationary Bandit Problems

Han Qi

QIHAN19@STU.XJTU.EDU.CN

Yue Wang

WY980521@STU.XJTU.EDU.CN

Li Zhu

ZHULI@XJTU.EDU.CN

*Xi'an Jiaotong University,  
School of Software Engineering,  
Xianning West Road 28,  
Xi'an, 710049, China*

## Abstract

Non-stationary multi-armed bandit (NS-MAB) problems have recently received significant attention. NS-MAB are typically modelled in two scenarios: abruptly changing, where reward distributions remain constant for a certain period and change at unknown time steps, and smoothly changing, where reward distributions evolve smoothly based on unknown dynamics. In this paper, we propose Discounted Thompson Sampling (DS-TS) with Gaussian priors to address both non-stationary settings. Our algorithm passively adapts to changes by incorporating a discounted factor into Thompson Sampling. DS-TS method has been experimentally validated, but analysis of the regret upper bound is currently lacking. Under mild assumptions, we show that DS-TS with Gaussian priors can achieve nearly optimal regret bound on the order of  $\tilde{O}(\sqrt{TB_T})$  for abruptly changing and  $\tilde{O}(T^\beta)$  for smoothly changing, where  $T$  is the number of time steps,  $B_T$  is the number of breakpoints,  $\beta$  is associated with the smoothly changing environment and  $\tilde{O}$  hides the parameters independent of  $T$  as well as logarithmic terms. Furthermore, empirical comparisons between DS-TS and other non-stationary bandit algorithms demonstrate its competitive performance. Specifically, when prior knowledge of the maximum expected reward is available, DS-TS has the potential to outperform state-of-the-art algorithms.

## 1. Introduction

The multi-armed bandit (MAB) problem is a well-known sequential decision problem. In each time step, the learner must choose an arm (referred to as an action) from a finite set of arms based on previous observations. The learner only receives the reward associated with the chosen action and does not observe the rewards of other unselected actions. The learner's goal is to maximize the expected cumulative reward over time or, alternatively, to minimize the regret incurred during the learning process. Regret is defined as the difference between the expected reward of the optimal arm (the arm with the highest expected reward) and the expected reward achieved by the MAB algorithm being used. Minimizing regret implies approaching the performance of the optimal arm as closely as possible.

MAB has found practical use in various scenarios, with one of the earliest applications being the diagnosis and treatment experiments proposed by Robbins (1952). In this experiment, each patient's treatment plan corresponds to an arm in the MAB problem, and the goal is to minimize the patient's health loss by making optimal treatment decisions.

{1}------------------------------------------------

Recently, MAB has gained wide-ranging applicability. For example, MAB algorithms have been used in online recommendation systems to improve user experiences and increase engagement (Li, Chu, Langford, & Wang, 2011; Bounineffoul, Bouzeghoub, & Ganarski, 2012; Li, Karatzoglou, & Gentile, 2016). Similarly, MAB has been employed in online advertising campaigns to optimize the allocation of resources and maximize the effectiveness of ad placements (Schwartz, Bradlow, & Fader, 2017). While the standard MAB model assumes fixed reward distributions, real-world scenarios often involve changing distributions over time. For instance, in online recommendation systems, the collected data gradually becomes outdated, and user preferences are likely to evolve (Wu, Iyer, & Wang, 2018). This dynamic nature necessitates the development of algorithms that can adapt to these changes, leading to the exploration of non-stationary MAB problems.

In recent years, extensive research has been conducted on non-stationary multi-armed bandit (MAB) problems. These research efforts can be broadly categorized into two approaches. The first category involves using change-point detection algorithms to identify when the reward distribution changes (Liu, Lee, & Shroff, 2018; Cao, Wen, Kveton, & Xie, 2019; Auer, Gajane, & Ortner, 2019; Chen, Lee, Luo, & Wei, 2019; Besson, Kaufmann, Maillard, & Seznec, 2022). The second category focuses on passively reducing the influence of past observations (Garivier & Moulines, 2011; Raj & Kalyani, 2017; Trovo, Paladino, Restelli, & Gatti, 2020; Baudry, Russac, & Cappé, 2021). The former approach relies on certain assumptions about the changes in the distribution of arms to ensure the effectiveness of the change-point detection algorithm. For example, methods proposed by Liu et al. (2018) and Cao et al. (2019) require a lower bound on the amplitude of change in the expected rewards for each arm. The latter approach requires fewer assumptions about the characteristics of the changes. These methods often employ techniques such as sliding windows or discount factors to forget past information and adapt to the changing distribution of arms. Frequentist algorithms in both categories provide theoretical guarantees for regret upper bounds. However, in the case of Bayesian methods, such as Thompson Sampling, there is a lack of theoretical analysis regarding regret in non-stationary MAB problems, despite these algorithms demonstrating superior or comparable performance to frequentist algorithms in most non-stationary scenarios. To the best of our knowledge, only sliding window Thompson Sampling (Trovo et al., 2020) has provided regret upper bounds. Raj and Kalyani (2017) have explored discounted Thompson Sampling with Bernoulli priors but only derived the probability of selecting a sub-optimal arm in the simple case of a two-armed bandit.

In this paper, we propose Discounted Thompson Sampling (DS-TS) with Gaussian priors for both abruptly changing and smoothly changing settings. In the former, the distributions of rewards remain constant during a period of rounds and change at unknown rounds, while in the latter, the reward distribution evolves smoothly based on unknown dynamics. We adopt a unified method to analyze the regret upper bound for both non-stationary settings. We show that the regret upper bound of DS-TS for abruptly changing settings is  $\tilde{O}(\sqrt{TB_T})$ , where  $T$  is the number of time steps,  $B_T$  is the number of breakpoints. This regret bound matches the  $\Omega(\sqrt{T})$  lower bound proven by Garivier and Moulines (2011) in an order sense. For the smoothly changing settings, we derive the regret bound of order  $\tilde{O}(T^\beta)$ , where  $\beta$  measures the number of rounds that the arms' expected rewards are close enough. In addition, we compare DS-TS with other non-stationary bandits algorithms empirically.

{2}------------------------------------------------

Specialy, if we know the information of the maximum of the expected rewards, by tuning the parameter  $\tau_{max}$ , our algorithm can outperform the state-of-the-art algorithms.

## 2. Related Work

Non-stationary MAB settings have received attention in the last few years. These methods can be roughly divided into two categories: they detect when the reward distribution changes with change-point detection algorithms or they passively reduce the effect of past observations. Most of these works can achieve the regret upper bound of  $\tilde{O}(\sqrt{T})$ .

Many works are based on the idea of forgetting past observations. Discounted UCB (DS-UCB) (Kocsis & Szepesvári, 2006; Garivier & Moulines, 2011) uses a discounted factor to average the past rewards. In order to achieve the purpose of forgetting information, the weight of the early reward is smaller. Garivier and Moulines (2011) also propose the sliding-window UCB (SW-UCB) by only using a few recent rewards to compute the UCB index. They calculate the regret upper bound for DS-UCB and SW-UCB as  $\tilde{O}(\sqrt{TB_T})$ . EXP3.S, as proposed in (Auer, Cesa-Bianchi, Freund, & Schapire, 2002), has been shown to achieve the regret upper bound by  $\tilde{O}(\sqrt{TB_T})$ . Under the assumption that the total variation of the expected rewards over the time horizon is bounded by a budget  $V_T$ , Besbes, Gur, and Zeevi (2014) introduce REXP3 with regret  $\tilde{O}(T^{2/3})$ . Combes and Proutiere (2014) propose the SW-OSUB algorithm, specifically for the case of smoothly changing with an upper bound of  $\tilde{O}(\sigma^{1/4}T)$ , where  $\sigma$  is the Lipschitz constant of the evolve process. Raj and Kalyani (2017) propose the discounted Thompson sampling for Bernoulli priors without providing the regret upper bound. They only calculate the probability of picking a sub-optimal arm for the simple case of a two-armed bandit. Recently, Trovo et al. (2020) propose the sliding-window Thompson sampling algorithm with regret  $\tilde{O}(T^{\frac{1+a}{2}})$  for abruptly changing and  $\tilde{O}(T^\beta)$  for smoothly changing. Baudry et al. (2021) propose a novel algorithm named Sliding Window Last Block Subsampling Duelling Algorithm (SW-LB-SDA) with regret  $\tilde{O}(\sqrt{TB_T})$ . They only assume that the reward distributions belong to the same one-parameter exponential family for all arms during each stationary phase.

There are also many works that exploit techniques from the field of change detection to deal with reward distributions varying over time. Mellor and Shapiro (2013) combine a Bayesian change point mechanism and Thompson sampling strategy to tackle the non-stationary problem. Their algorithm can detect global switching and per-arm switching. Liu et al. (2018) propose a change-detection framework that combines UCB and a change-detection algorithm named CUSUM. They obtain an upper bound for the average detection delay and a lower bound for the average time between false alarms. Cao et al. (2019) propose M-UCB, which is similar to CUSUM but use another simpler change-detection algorithm. M-UCB and CUMSUM are nearly optimal, their regret bounds are  $\tilde{O}(\sqrt{TB_T})$ .

Recently, there are also some works deriving regret bounds without knowing the number of changes. For example, Auer et al. (2019) propose an algorithm called ADSWITCH with optimal regret bound  $\tilde{O}(\sqrt{B_T T})$ . Suk and Kpotufe (2022) improve the work (Auer et al., 2019) so that the obtained regret bound is smaller than  $\tilde{O}(\sqrt{ST})$ , where  $S$  only counts the best arms switches.

{3}------------------------------------------------

## 3. Problem Formulation

Assume that the non-stationary MAB problem has  $K$  arms  $\mathcal{A} := \{1, 2, \dots, K\}$  with finite time horizon  $T$ . At round  $t$ , the learner must select an arm  $i_t \in \mathcal{A}$  and obtain the corresponding reward  $X_t(i_t)$ . The rewards are generated from different distributions (unknown to the learner) with bounded support. Without loss of generality, suppose the support set is  $[0, 1]$ . The expectation of  $X_t(i)$  is denoted as  $\mu_t(i) = \mathbb{E}[X_t(i)]$ . A policy  $\pi$  is a function  $\pi(i_t) = i_t$  that selects arm  $i_t$  to play at round  $t$ . Let  $\mu_t(*) := \max_{i \in \{1, \dots, K\}} \mu_t(i)$  denote the expected reward of the optimal arm  $i_t^*$  at round  $t$ . Unlike the stationary MAB settings, where an arm is optimal all of the time (i.e.  $\forall t \in \{1, \dots, T\}, i_t^* = i^*$ ), while in the non-stationary settings, the optimal arms might change over time. The performance of a policy  $\pi$  is measured in terms of cumulative expected regret:

$$R_T^\pi = \mathbb{E} \left[ \sum_{t=1}^{T} (\mu_t(*) - \mu_t(i_t)) \right], \quad (1)$$

where  $\mathbb{E}[\cdot]$  is the expectation with respect to randomness of  $\pi$ . Let  $\Delta_t(i) := \mu_t(*) - \mu_t(i)$  and let

$$k_T(i) := \sum_{t=1}^{T} \mathbb{1}\{i_t = i, i \neq i_t^*\}$$

denote the number of plays of arm  $i$  when it is not the best arm until time  $T$ ,

$$R_T^\pi = \sum_{i=1}^{K} \sum_{t=1}^{T} \Delta_t(i) \mathbb{E}[\mathbb{1}\{i_t = i\}] \le \sum_{i=1}^{K} \mathbb{E}[k_T(i)].$$

When we analyze the upper bound of  $R_T^\pi$ , we can directly analyze  $\mathbb{E}[k_T(i)]$  to get the upper bound of each arm. Next, we give detailed description of the two non-stationary scenarios.

**Abruptly Changing.** The abruptly changing settings is introduced by Garivier and Moulines (2011) for the first time. Suppose the set of *breakpoints* is  $\mathcal{B} = \{b_1, \dots, b_{B_T}\}$  (we define  $b_1 = 1$ ). At each breakpoint, the reward distribution changes for at least one arm. The rounds between two adjacent breakpoints are called *stationary phase*. In the stationary phase, the reward distribution of all arms does not change. Different from previous studies (Besbes et al., 2014; Liu et al., 2018; Cao et al., 2019), which imposed constraints on the variation of expected rewards, we do not impose constraints on this variation in our settings. Trovo et al. (2020) makes assumption about the number of breakpoints to facilitate more generalized analysis, while we explicitly use  $B_T$  to represent the number of breakpoints for analysis.

**Smoothly Changing.** The smoothly changing setting have been studied by Combes and Proutiere (2014), Trovo et al. (2020). At each time step, the expected reward for each arm varies by no more than  $\sigma$  and the learner doesn't have any information on how the rewards evolve. These limitations can be described by the following Lipschitz assumption:

**Assumption 1.** There exists  $\sigma > 0$ , for all  $t, t' \ge 1, 1 \le i \le K$ , it holds that  $|\mu_t(i) - \mu_{t'}(i)| \le \sigma |t - t'|$ .

{4}------------------------------------------------

## 4. Discounted Thompson Sampling

In this section, we propose the Discounted Thompson Sampling algorithm with Gaussian priors for the non-stationary stochastic MAB problems. As in (Agrawal & Goyal, 2013), our algorithm uses an implicit assumption that the likelihood of reward  $X_i(t)$  can be modeled by Gaussian distribution. While the actual rewards distribution can be any distribution with support in  $[0, 1]$ . We use a discount factor  $\gamma$  ( $0 < \gamma < 1$ ) to dynamically adjust the estimate of each arm's distribution. The key to our algorithm is to decrease the sampling variance of the selected arm while increasing the sampling variance of the unselected arms.

Specifically, let  $N_t(\gamma, i) = \sum_{j=1}^t \gamma^{t-j} \mathbb{1}\{i_j = i\}$  denotes the discounted number of plays of arm  $i$  until time  $t$ . We use  $\hat{\mu}_t(\gamma, i) = \frac{1}{N_t(\gamma, i)} \sum_{j=1}^t \gamma^{t-j} X_j(i) \mathbb{1}\{i_j = i\}$  called discounted empirical average to estimate the expected rewards of arm  $i$ . The sampling variance for arm  $i$  at round  $t$  is denoted as  $\tau_t(i)^2$ . At round  $t$ , arm  $i$  samples from the Gaussian distribution  $\mathcal{N}(\hat{\mu}_t(\gamma, i), \tau_t(i)^2)$ . Recall that the rewards are in range  $[0, 1]$ , thus the variance cannot be increased to infinity, which would move the sampling away from the expectation. We restrict the upper bound on the variance to  $\tau_{max}^2$ , then the sampling variance is  $\tau_t(i) = \min\{\sqrt{\frac{1}{N_t(\gamma, i)}}, \tau_{max}\}$ .

Let  $\hat{\mu}_t(\gamma, i) = \sum_{j=1}^t \gamma^{t-j} X_j(i) \mathbb{1}\{i_j = i\}$  as the discounted cumulative reward. If arm  $i$  is selected at round  $t$ , the posterior distribution is updated as follows:

$$\hat{\mu}_{t+1}(\gamma, i) = \frac{\gamma \hat{\mu}_t(\gamma, i) N_t(\gamma, i) + X_t(i)}{\gamma N_t(\gamma, i) + 1} = \frac{\hat{\mu}_t(\gamma, i)}{N_{t+1}(\gamma, i)}$$

If arm  $i$  isn't selected at round  $t$ , the posterior distribution is updated as

$$\hat{\mu}_{t+1}(\gamma, i) = \frac{\hat{\mu}_t(\gamma, i)}{N_{t+1}(\gamma, i)} = \frac{\gamma \hat{\mu}_t(\gamma, i)}{\gamma N_t(\gamma, i)} = \hat{\mu}_t(\gamma, i)$$

i.e. the expectation of posterior distribution remains unchanged.

Algorithm 1 shows the pseudocode of DS-TS. We initialize the prior distributions with  $\hat{\mu}_1(i) = 0, \tau_1(i) = 1$ . Line 5 is the Thompson sampling. For each arm, we draw a random sample  $\theta_t(i)$  from  $\mathcal{N}(\hat{\mu}_t(\gamma, i), \tau_t(i)^2)$ . Then we select arm  $i_t$  with the maximum sample value to play and obtain the reward  $X_t(i_t)$  (Line 7). To avoid the time complexity going to  $O(T^2)$ , we introduce  $\tilde{\mu}_t(i)$  to calculate  $\hat{\mu}_t(\gamma, i)$  using an iterative method (Line 9-11). Finally, we update the posterior variance  $\tau_t^2$  of each arm (Line 12).

**Related to Thompson Sampling.** If  $\gamma = 1$ , DS-TS is equivalent to Thompson Sampling with stationary settings proposed by Agrawal and Goyal (2013) except for the variance. Their sampling variance is  $\frac{1}{k_i+1}$  ( $k_i$  is the number plays of arm  $i$ ), which are updated according to the standard Bayesian posterior distribution. While we truncate it to  $\tau_{max}$  to prevent the posterior variance from becoming infinite.

**Related to Discounted UCB.** Line 7 in Algorithm 1 can be rewritten as

$$i_t = \arg \max_i \hat{\mu}_t(\gamma, i) + \epsilon_t(i), \epsilon_t(i) \sim \mathcal{N}(0, \tau_t(i)^2)$$

We use the same method as DS-UCB (Garivier & Moulines, 2011) to update  $\hat{\mu}_t(\gamma, i)$ . As for selecting the best arm, DS-UCB uses a padding function  $c_t(\gamma, i) = 2B \sqrt{\frac{\xi \log n_t(\gamma)}{N_t(\gamma, i)}}$ , where  $B, \xi$

{5}------------------------------------------------

---

**Algorithm 1:** DS-TS

---

```
1 Input: discounted factor  $\gamma \in (1 - \frac{1}{e}, 1)$ ,  $\tau_{max}$ ,
2  $\hat{\mu}_1(i) = 0$ ,  $\hat{\mu}_1(i) = 0$ ,  $N_t(\gamma, i) = 0$ ,  $\tau_t(i) = \tau_{max}$ .
3 for  $t = 1, \dots, T$  do
4   for  $i = 1, \dots, K$  do
5     | sample  $\theta_t(i)$  independently from  $\mathcal{N}(\hat{\mu}_t(\gamma, i), \tau_t(i)^2)$ 
6   end
7   Play arm  $i_t = \arg\max_i \theta_t(i)$  and observe reward  $X_t(i_t)$ .
8   for  $i = 1, \dots, K$  do
9     |  $\hat{\mu}_{t+1}(\gamma, i) = \gamma \hat{\mu}_t(\gamma, i) + \mathbb{1}\{i = i_t\} X_t(i_t)$ 
10    |  $N_{t+1}(\gamma, i) = \gamma N_t(\gamma, i) + \mathbb{1}\{i = i_t\}$ 
11    |  $\hat{\mu}_{t+1}(\gamma, i) = \frac{\hat{\mu}_{t+1}(\gamma, i)}{N_{t+1}(\gamma, i)}$ 
12    |  $\tau_{t+1}(i) = \min\left\{\frac{1}{\sqrt{N_{t+1}(\gamma, i)}}, \tau_{max}\right\}$ 
13   end
14 end
```

---

is the tuning parameters,  $n_t(\gamma) = \sum_{i=1}^K N_t(\gamma, i)$ . While our approach can be viewed as replacing the padding function with  $\epsilon_t(i)$ . Sampling from the normal distribution  $\mathcal{N}(0, \tau_t(i)^2)$  is more exploratory than using deterministic padding function. In the experiment section, we will show the advantages of DS-TS.

## 5. Our Results

In this section, we give the upper bounds of DS-TS with abruptly changing and smoothly changing settings. Then we discuss how to take the values of the parameters so that the DS-TS reaches the optimal upper bound.

### 5.1 Abruptly Changing Settings

Recall that  $\Delta_t(i) := \mu_t(*) - \mu_t(i)$ . Let  $\Delta_T = \min\{\Delta_t(i) : t \le T, i \neq i_t^*\}$ , be the minimum difference between the expected reward of the best arm  $i_t^*$  and the expected reward of all arm in all time  $T$  when the arm is not the best arm, and  $\mu_{max} = \max_{t \in \{1, \dots, T\}, i \in \{1, \dots, K\}} \mu_t(i) \in (0, 1)$ , be the maximum of expected rewards. Define function  $F(x) = \frac{1}{\sqrt{2\pi}} \frac{x}{1+x^2} e^{-x^2/2}$ .

**Theorem 1.** Let  $\gamma \in (1 - \frac{1}{e}, 1)$ ,  $\tau_{max} \ge \frac{1}{12\sqrt{2}}$ . In the abruptly changing settings, for any arm  $i \in \{1, \dots, K\}$ ,

$$\mathbb{E}[k_T(i)] \le B_T D(\gamma) + (C + 2)L(\gamma) \gamma^{-1/(1-\gamma)} T (1 - \gamma) \log\left(\frac{1}{1 - \gamma}\right),$$

$$\text{where } D(\gamma) = \frac{\log((1-\gamma)^2 \log(1-\gamma))}{\log \gamma}, L(\gamma) = \frac{144(1+\sqrt{2})^2 \log(\frac{1}{1-\gamma} + e^{25})}{\gamma^{1/(1-\gamma)} (\Delta_T)^2}, C = e^{25} + 12 + \frac{1}{F(\frac{\mu_{max}}{\tau_{max}})}.$$

**Corollary 1.** When  $\gamma$  is close to 1,  $\gamma^{-1/(1-\gamma)}$  is around  $e$ . If the time horizon  $T$  and number of breakpoints  $B_T$  are known in advance, the discounted factor can be chosen as  $\gamma = 1 -$

{6}------------------------------------------------

$\sqrt{B_T/T}$ , then  $\mathbb{E}[k_T(i)] = O(\sqrt{T B_T} \log^2(T))$ . If  $B_T = O(T^\alpha)$  for some  $\alpha \in (0, 1)$ , this regret is upper bounded by  $O(T^{\frac{1+\alpha}{2}} \log^2(T))$ .

**Remark 1.** The condition  $\tau_{\max} \ge \frac{1}{12\sqrt{2}}$  is imposed to help the analysis. From the expression for  $F(x)$  it is obvious that  $\tau_{\max}$  has a lower bound greater than 0. In fact, follows from the proof of Lemma 2,  $\tau_{\max}$  needs to satisfy the condition  $\tau_{\max} \ge \frac{\Delta_T}{12\sqrt{2+3\sqrt{1-\gamma}}} \frac{1}{\sqrt{\log \frac{1}{1-\gamma}}}$ .

Since  $\gamma > 1 - \frac{1}{e}$ ,  $\tau_{\max} \ge \frac{1}{12\sqrt{2}}$  is clearly satisfied. Combining practical experience, too small sampling variance will make Thompson Sampling lose its exploration ability. In addition, if we know  $\mu_{\max}$  in advance,  $\tau_{\max}$  can be set suitably to improve the empirical performance of DS-TS. In the experimental section we will describe how to take the appropriate  $\tau_{\max}$ .

### 5.2 Smoothly Changing Settings

Smoothly changing settings present greater challenges compared to abruptly changing settings. The primary difficulty arises from the fact that the expected rewards of different arms may be extremely close to each other. In order to effectively address this challenge, we require certain assumption that restricts the number of rounds in which the expected rewards of two arms can become arbitrarily small. This assumption is necessary to ensure that the arms' expected rewards remain distinguishable and prevent the algorithm from being overwhelmed by the inherent uncertainty in the rewards. Let  $\Delta \in (0, 1)$ , define

$$H(\Delta, T) = \{t \in \{1, \dots, T\} : \exists i \neq j, |\mu_i(t) - \mu_j(t)| < \Delta\}$$

and we make the following assumption.

**Assumption 2.** There exist some constant independent of  $T$ ,  $\beta \in [0, 1]$ , positive number  $F$ ,  $\Delta_0 \in (0, 1)$ , s.t. for all  $\Delta < \Delta_0$ ,

$$|H(\Delta, T)| \le F \Delta T^\beta$$

This assumption is consistent with (Trovo et al., 2020), and if  $\beta = 1$  it's equivalent to that in (Combes & Proutiere, 2014).

**Theorem 2.** Let  $\gamma \in (1 - \frac{1}{e}, 1)$ ,  $\tau_{\max} \ge \frac{1}{12\sqrt{2}}$  Lipschitz constant  $\sigma > 0$ . There exists  $\Delta_0$  as in Assumption 2, s.t.  $2\sigma D(\gamma) < \Delta/3 \le \Delta_0$ . In the smoothly changing settings, for any arm  $i \in \{1, \dots, K\}$ ,

$$\mathbb{E}[k_T(i)] \le F \Delta T^\beta + M(\gamma) T (1 - \gamma) \log\left(\frac{1}{1 - \gamma}\right), \quad (2)$$

$$\text{where } M(\gamma) = \frac{144(1+\sqrt{2})^2 \log(\frac{1}{1-\gamma} + e^{25})}{\gamma^{1/(1-\gamma)} \Delta^2} (e^{25} + 13 + \frac{1}{F(\frac{\tau_{\max}}{1-\gamma})}) + \frac{594}{\gamma^{1/(1-\gamma)} (\Delta/3 - 2\sigma D(\gamma))^2}.$$

**Corollary 2.** If  $\beta$  in Assumption 2 and the time horizon  $T$  are known in advance, the discounted factor can be set as  $\gamma = 1 - 1/T^{1-\beta}$ , then  $\mathbb{E}[k_T(i)] = O(T^\beta \log^2(T))$ .

**Remark 2.** Trovo et al. (2020) discusses in detail the value of  $\beta$  for which Assumption 2 holds under the conditions of Theorem 2 and Corollary 2. Define

$$P = |\{t \in \{1, \dots, T-1\} : \exists i \neq j, (\mu_i(t) - \mu_j(t))(\mu_i(t+1) - \mu_j(t+1)) < 0\}|$$

{7}------------------------------------------------

as the number of times the expected rewards of a pair of arms change over the time period. When  $\gamma = 1 - 1/T^{1-\beta}$ , it's easy to get  $D(\gamma) \le 2T^{1-\beta} \log T$ . We can get  $\beta$  need within  $[\max\{1 - \log_T(\frac{\Delta}{12\sigma \log T}), \frac{1}{2} - \log_T \sqrt{\frac{F\Delta}{24P \log T}}, 1\}]$  in our setting.

## 6. Proofs of Upper Bounds

In this section, we prove the results respectively. The proofs of the two settings follow a similar approach. The differences in the proof can be dealt with uniformly using Lemma 1 and Lemma 3. The core framework of our proof follows (Agrawal & Goyal, 2013). We extend the analysis of (Agrawal & Goyal, 2013) to the non-stationary case by combining the method proposed in (Garivier & Moulines, 2011). The main difficulty lies in the need to estimate the  $\mathbb{E}[\frac{1}{p_{i,t}}]$  for non-stationary settings (Lemma 2 and Lemma 4).

### 6.1 Proofs of Theorem 1

**Proof outlines:** The idea is to divide  $\mathbb{E}[k_T(i)]$  into several parts according to whether the specific events are true or not. First of all the expected rewards for all arms will not be well estimated in the rounds near the breakpoints, and this part can be bounded as  $B_T D(\gamma)$ . Then focus on the rounds that are far from the breakpoints ( $D(\gamma)$  rounds after the breakpoints). For the rounds in which the mean of the arm is not well estimated, the regret can be bounded by Lemma 6. For the rounds that the arm is fully explored, the regret bound can be estimated by self-normalized Hoeffding-type inequality (Garivier & Moulines, 2011). For the other cases, one can use Lemma 5 and thus needs to estimate  $\mathbb{E}[\frac{1}{p_{i,t}}]$ . We derive the upper bound of  $\mathbb{E}[\frac{1}{p_{i,t}}]$  for non-stationary settings, with an extra logarithmic term compared with the stationary settings.

Before proceeding to the specific analysis, we first give some definitions and lemmas that will be used in both non-stationary settings.

**Definition 1 (Quantities  $x_t(i), y_t(i)$ ).** For arm  $i \neq i_t^*$ , we choose two threshold  $x_t(i), y_t(i)$  such that  $x_t(i) = \mu_t(i) + \frac{\Delta_t(i)}{3}, y_t(i) = \mu_t(*) - \frac{\Delta_t(i)}{3}$ . Then  $\mu_t(i) < x_t(i) < y_t(i) < \mu_t(*)$  and  $y_t(i) - x_t(i) = \frac{\Delta_t(i)}{3}$ .

**Definition 2 ( $\tilde{\mu}_t(\gamma, i)$ ).**  $\tilde{\mu}_t(\gamma, i) = \frac{1}{N_t(\gamma, i)} \sum_{j=1}^t \gamma^{t-j} \mathbb{1}\{i_j = i\} \mu_j(i)$  denotes the discounted average of expectation for arm  $i$  at time step  $t$ . If the randomness of  $\mathbb{1}\{i_j = i\}$  is removed, i.e., the selection of each arm is deterministic, then  $\tilde{\mu}_t(\gamma, i) = \mathbb{E}[\tilde{\mu}_t(\gamma, i)]$ .

**Definition 3 (History  $\mathcal{F}_t$ ).**  $\mathcal{F}_t$  is the sequence

$$\mathcal{F}_t = \{i_k, X_k(i_k), k = 1, \dots, t\},$$

where  $i_k$  denotes the arm played at time  $k$ , and  $X_k(i_k)$  denotes the reward obtained at time  $k$ . Define  $\mathcal{F}_0 = \{\}$ .

By definition, we know  $\mathcal{F}_0 \subseteq \mathcal{F} \subseteq \dots \subseteq \mathcal{F}_T$ . And  $i_t, \tilde{\mu}_t(\gamma, i)$ , the distribution of  $\theta_t(i)$  are determined by the history  $\mathcal{F}_{t-1}$ .

Now we give some additional definitions and lemmas for abruptly changing settings only. The abruptly changing settings is in fact piecewise-stationary. Some rounds between

{8}------------------------------------------------

two breakpoints appear to be stationary. Based on this observation, we give the following definition.

**Definition 4 (Pseudo-Stationary Phase  $\mathcal{T}(\gamma)$ ).**  $\mathcal{T}(\gamma) = \{t \le T : \forall s \in (t-D(\gamma), t], \mu_s(\cdot) = \mu_t(\cdot)\}$ , where  $D(\gamma) = \log((1-\gamma)^2 \log(\frac{1}{1-\gamma})) / \log \gamma$ .

**Remark 3.** Let  $\mathcal{S}(\gamma) = \{t \le T : t \notin \mathcal{T}(\gamma)\}$ . Note that, on the right side of any breakpoint, there will be  $D(\gamma)$  rounds belonging to  $\mathcal{S}(\gamma)$ . Therefore, the number of elements in the set  $\mathcal{S}(\gamma)$  has an upper bound  $B_T D(\gamma)$ , i.e.  $|\mathcal{S}(\gamma)| \le B_T D(\gamma)$ .  $\mathcal{T}(\gamma)$  is called the pseudo-stationary phase because the length of  $\mathcal{T}(\gamma)$  is smaller than the true stationary phase. Figure 1 shows  $\mathcal{T}(\gamma)$  and  $\mathcal{S}(\gamma)$  in two different situations.

![Figure 1: Illustration of T(gamma) and S(gamma) in two different situations. The top figure shows a timeline with breakpoints b_i and b_{i+1}. The interval [b_i, b_i + D(gamma)] is labeled S(gamma) and the interval [b_i + D(gamma), b_{i+1}] is labeled T(gamma). The condition b_{i+1} - b_i > D(gamma) is shown. The bottom figure shows a similar timeline but with b_{i+1} - b_i <= D(gamma), where the T(gamma) interval overlaps with the S(gamma) interval of the next breakpoint.](bedcca5cdf168e3508ef511d94ec514c_img.jpg)

Figure 1: Illustration of T(gamma) and S(gamma) in two different situations. The top figure shows a timeline with breakpoints b\_i and b\_{i+1}. The interval [b\_i, b\_i + D(gamma)] is labeled S(gamma) and the interval [b\_i + D(gamma), b\_{i+1}] is labeled T(gamma). The condition b\_{i+1} - b\_i > D(gamma) is shown. The bottom figure shows a similar timeline but with b\_{i+1} - b\_i <= D(gamma), where the T(gamma) interval overlaps with the S(gamma) interval of the next breakpoint.

Figure 1: Illustration of  $\mathcal{T}(\gamma)$  and  $\mathcal{S}(\gamma)$  in two different situations.  $b_{i+1} - b_i > D(\gamma)$  is shown in the top figure, and  $b_{i+1} - b_i \le D(\gamma)$  the bottom figure.

Unlike the Sliding window method (Garivier & Moulines, 2011; Trovo et al., 2020),  $\mu_t(i)$  and  $\tilde{\mu}_t(\gamma, i)$  are often not the same even in the pseudo-stable phase  $\mathcal{T}(\gamma)$ . The following lemma depicts that after finite rounds at the breakpoint, that is, in the pseudo-stable phase, the distance between  $\mu_t(i)$  and  $\tilde{\mu}_t(\gamma, i)$  has an upper bound.

**Lemma 1.**  $\forall t \in \mathcal{T}(\gamma)$ , the distance between  $\mu_t(i)$  and  $\tilde{\mu}_t(\gamma, i)$  is less than  $U_t(\gamma, i)$ .

$$|\mu_t(i) - \tilde{\mu}_t(\gamma, i)| \le U_t(\gamma, i), \quad (3)$$

where

$$U_t(\gamma, i) = \sqrt{\frac{(1-\gamma) \log \frac{1}{1-\gamma}}{N_t(\gamma, i)}}.$$

As can be seen from Definition 4,  $(t - D(\gamma), t)$  does not contain any breakpoints  $\Leftrightarrow t \in \mathcal{T}(\gamma)$ . For any breakpoint  $b_i \in \{b_1, \dots, b_{B_T}\}$ ,  $b_i + D(\gamma) \in \mathcal{T}(\gamma)$  if  $D(\gamma) \le b_{i+1} - b_i$ . That is,  $D(\gamma)$  rounds after the breakpoint  $b_i$  ( $D(\gamma) \le b_{i+1} - b_i$ ), the distance between  $\mu_t(i)$  and  $\tilde{\mu}_t(\gamma, i)$  is less than  $U_t(\gamma, i)$ .

**Lemma 2.** Let  $p_{i,t} = \mathbb{P}(\theta_t(*) > y_t(i) | \mathcal{F}_{t-1})$ . For any  $t \in \mathcal{T}(\gamma)$  and  $i \neq i_t^*$ ,

$$\sum_{t \in \mathcal{T}(\gamma)} \mathbb{E} \left[ \frac{1 - p_{i,t}}{p_{i,t}} \mathbb{1}\{i_t = i_t^*, \theta_t(i) < y_t(i)\} \right] \le CT(1-\gamma)L(\gamma)\gamma^{-1/(1-\gamma)} \log \frac{1}{1-\gamma}.$$

{9}------------------------------------------------

$$\text{where } C = e^{25} + \frac{1}{F(\frac{1}{\tau_{\max}})} + 12, L(\gamma) = \frac{144(1+\sqrt{2})^2 \log(\frac{1}{1-\gamma} + e^{25})}{\Delta_T^2}.$$

To facilitate the analysis, we define some quantities that are independent of  $t$ .

$$m = \frac{12\sqrt{2}}{\sqrt{1-\gamma}} + 3, n = 12\sqrt{2} + 3\sqrt{1-\gamma}, A(\gamma) = \frac{n^2 \log(\frac{1}{1-\gamma})}{(\Delta_T)^2},$$

From the definition of  $U_t(\gamma, i)$  given in Lemma 1, we can get

$$U_t(\gamma, i) = \frac{\Delta_T}{m} \sqrt{\frac{A(\gamma)}{N_t(\gamma, i)}}. \quad (4)$$

Now we can give the detailed proof. The proof is in 5 steps:

**Step 1** We can divide the rounds  $t \in \{1, \dots, T\}$  into two parts:  $\{t \in \mathcal{T}(\gamma)\}$  and  $\{t \notin \mathcal{T}(\gamma)\}$ . From Remark 3, the number of elements in the second part is smaller than  $B_T D(\gamma)$ .

$$\mathbb{E}[k_T(i)] \le B_T D(\gamma) + \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(i_t = i). \quad (5)$$

**Step 2** Then we consider the event  $\{N_t(\gamma, i) > A(\gamma)\}$ .

$$\sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(i_t = i) = \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(i_t = i, N_t(\gamma, i) < A(\gamma)) + \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(i_t = i, N_t(\gamma, i) > A(\gamma)).$$

We first bound  $\sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(i_t = i, N_t(\gamma, i) < A(\gamma))$ .

$$\begin{aligned} \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(i_t = i, N_t(\gamma, i) < A(\gamma)) &= \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E}[\mathbb{P}(i_t = i, N_t(\gamma, i) < A(\gamma)) | \mathcal{F}_{t-1}] \\ &= \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E}[\mathbb{E}[\mathbb{1}\{i_t = i, N_t(\gamma, i) < A(\gamma)\} | \mathcal{F}_{t-1}]] \\ &\stackrel{(a)}{=} \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E}[\mathbb{1}\{i_t = i, N_t(\gamma, i) < A(\gamma)\}] \\ &\stackrel{(b)}{\le} T(1-\gamma)A(\gamma)\gamma^{-1/(1-\gamma)} \end{aligned} \quad (6)$$

where (a) uses the tower rule for expectation, (b) follows from Lemma 6. Therefore,

$$\mathbb{E}[k_T(i)] \le T(1-\gamma)A(\gamma)\gamma^{-1/(1-\gamma)} + B_T D(\gamma) + \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(i_t = i, N_t(\gamma, i) > A(\gamma)) \quad (7)$$

**Step 3** Define  $E_t(\gamma, i)$  as the event that  $\{i_t = i, N_t(\gamma, i) > A(\gamma)\}$ . Define  $E_t^{\theta}(i)$  as the event that  $\theta_t(i) < y_t(i)$ . This part may be decomposed as follows:

$$\begin{aligned} \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i)) &= \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) > x_t(i)) + \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i), \overline{E_t^{\theta}(i)}) \\ &\quad + \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i), E_t^{\theta}(i)) \end{aligned} \quad (8)$$

{10}------------------------------------------------

Next, we bound the first part by Lemma 7.

$$\begin{aligned}
& \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) > x_t(i)) \\
& \le \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(\hat{\mu}_t(\gamma, i) > \mu_t(i) + \frac{\Delta_t(i)}{3}, N_t(\gamma, i) > A(\gamma)) \\
& \le T(1 - \gamma)^{48} \log \frac{1}{1 - \gamma}
\end{aligned} \tag{9}$$

**Step 4** Then we bound the second part.

$$\begin{aligned}
& \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i), \overline{E}_t^\theta(i)) \\
& = \mathbb{E} \left[ \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E}[\mathbb{1}\{i_t = i, N_t(\gamma, i) > A(\gamma), \hat{\mu}_t(\gamma, i) < x_t(i), \overline{E}_t^\theta(i)\} | \mathcal{F}_{t-1}] \right] \\
& \stackrel{(c)}{=} \mathbb{E} \left[ \sum_{t \in \mathcal{T}(\gamma)} \mathbb{1}\{N_t(\gamma, i) > A(\gamma), \hat{\mu}_t(\gamma, i) < x_t(i)\} \mathbb{P}(i_t = i, \overline{E}_t^\theta(i) | \mathcal{F}_{t-1}) \right] \\
& \le \mathbb{E} \left[ \sum_{t \in \mathcal{T}(\gamma)} \mathbb{1}\{N_t(\gamma, i) > A(\gamma), \hat{\mu}_t(\gamma, i) < x_t(i)\} \mathbb{P}(\theta_t(i) > y_t(i) | \mathcal{F}_{t-1}) \right],
\end{aligned} \tag{10}$$

where (c) uses the fact that  $N_t(\gamma, i)$  and  $\hat{\mu}_t(i)$  are determined by the history  $\mathcal{F}_{t-1}$ . Therefore, given the history  $\mathcal{F}_{t-1}$  such that  $N_t(\gamma, i) > A(\gamma)$  and  $\hat{\mu}_t(\gamma, i) < x_t(i)$ , we have

$$\mathbb{P}(\theta_t(i) > y_t(i) | \mathcal{F}_{t-1}) \le \mathbb{P}(\theta_t(i) - \hat{\mu}_t(\gamma, i) > \frac{\Delta_T}{3} | \mathcal{F}_{t-1}) \le \frac{1}{2} \exp\left(-\frac{(\Delta_T)^2 A(\gamma)}{18}\right) \le \frac{1}{2} (1 - \gamma)^{16}.$$

For other  $\mathcal{F}_{t-1}$ , the indicator term  $\mathbb{1}\{N_t(\gamma, i) > A(\gamma), \hat{\mu}_t(\gamma, i) < x_t(i)\}$  will be 0. Hence, we can bound the second part by

$$\sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i), \overline{E}_t^\theta(i)) \le \frac{T}{2} (1 - \gamma)^{16}$$

**Step 5** Finally, using Lemma 2,

$$\begin{aligned}
\sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i), E_t^\theta(i)) & \le \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E} \left[ \frac{1 - p_{i,t}}{p_{i,t}} \mathbb{P}(i_t = i_t^*, E_t^\theta(i) | \mathcal{F}_{t-1}) \right] \\
& \stackrel{(d)}{=} \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E} \left[ \mathbb{E} \left[ \frac{1 - p_{i,t}}{p_{i,t}} \mathbb{1}\{i_t = i_t^*, E_t^\theta(i)\} | \mathcal{F}_{t-1} \right] \right] \\
& = \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E} \left[ \frac{1 - p_{i,t}}{p_{i,t}} \mathbb{1}\{i_t = i_t^*, E_t^\theta(i)\} \right] \\
& \le CT(1 - \gamma)L(\gamma)\gamma^{-1/(1-\gamma)} \log \frac{1}{1 - \gamma}.
\end{aligned}$$

where (d) uses the fact that  $p_{i,t}$  is fixed given  $\mathcal{F}_{t-1}$ ,

{11}------------------------------------------------

Substituting the results in Step 3-5 to Equation (8) and Equation (7),

$$\begin{aligned}
\mathbb{E}[k_T(i)] &\le T(1-\gamma)A(\gamma)\gamma^{-1/(1-\gamma)} + B_T D(\gamma) + \sum_{t \in \mathcal{T}(\gamma)} \mathbb{P}(\dot{i}_t = i, N_t(\gamma, i) > A(\gamma)) \\
&\le T(1-\gamma)A(\gamma)\gamma^{-1/(1-\gamma)} + B_T D(\gamma) + T(1-\gamma) \log \frac{1}{1-\gamma} \\
&\quad + CT(1-\gamma)L(\gamma)\gamma^{-1/(1-\gamma)} \log \frac{1}{1-\gamma} \\
&\le B_T D(\gamma) + (C+2)L(\gamma)\gamma^{-1/(1-\gamma)}T(1-\gamma) \log \frac{1}{1-\gamma}.
\end{aligned}$$

### 6.2 Proofs of Theorem 2

The proof of Theorem 2 is similar to Theorem 1. The main difference is that there is no pseudo-stationary phase under smoothly changing settings. Fortunately, conclusions (Lemma 3, Lemma 4) similar to Lemma 1 and Lemma 2 still hold.

Recall that

$$\begin{aligned}
U_t(\gamma, i) &= \sqrt{\frac{(1-\gamma) \log \frac{1}{1-\gamma}}{N_t(\gamma, i)}}, D(\gamma) = \frac{\log((1-\gamma)^2 \log(\frac{1}{1-\gamma}))}{\log \gamma}, \\
m &= \frac{12\sqrt{2}}{\sqrt{1-\gamma}} + 3, n = 12\sqrt{2} + 3\sqrt{1-\gamma}
\end{aligned}$$

**Lemma 3.** For any  $t$  and  $\sigma$  satisfies Assumption 1,

$$|\mu_t(i) - \bar{\mu}_t(\gamma, i)| \le U_t(\gamma, i) + \sigma D(\gamma), \quad (11)$$

**Lemma 4.** Let  $p_{i,t} = \mathbb{P}(\theta_t(*) > y_t(i) - \sigma D(\gamma) | \mathcal{F}_{t-1})$ . For any  $i \neq i_t^*$ ,

$$\sum_{t=1}^{T} \mathbb{E} \left[ \frac{1 - p_{i,t}}{p_{i,t}} \mathbb{1}_{\{i_t = i_t^*, \theta_t(i) < y_t(i) - \sigma D(\gamma)\}} \right] \le CT(1-\gamma)L(\gamma)\gamma^{-1/(1-\gamma)} \log \frac{1}{1-\gamma}.$$

$$\text{where } C = e^{25} + \frac{1}{F \Gamma \frac{\mu_{\max}}{\tau_{\max}}} + 12, L(\gamma) = \frac{144(1+\sqrt{2})^2 \log(\frac{1}{1-\gamma} + e^{25})}{\Delta^2}.$$

We redefine  $A(\gamma)$  in smoothly changing settings as

$$A(\gamma) = \frac{n^2 \log(\frac{1}{1-\gamma})}{(\Delta/3 - 2\sigma D(\gamma))^2}.$$

Since there is no pseudo-stationary phase, the proof is only need divide into four steps.

**Step 1** The first step is exactly the same as the step 2 of Theorem 1, so we can get the following directly:

$$\mathbb{E}[k_T(i)] \le F\Delta T^\beta + T(1-\gamma)A(\gamma)\gamma^{-1/(1-\gamma)} + \sum_{t=1}^{T} \mathbb{P}(\dot{i}_t = i, N_t(\gamma, i) > A(\gamma)) \quad (12)$$

{12}------------------------------------------------

**Step 2** Define  $E_t(\gamma, i)$  as the event that  $\{i_t = i, N_t(\gamma, i) > A(\gamma)\}$ . Then

$$\begin{aligned} \sum_{t=1}^{T} \mathbb{P}(E_t(\gamma, i)) &= \sum_{t=1}^{T} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) > x_t(i) + \sigma D(\gamma)) \\ &\quad + \sum_{t=1}^{T} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i) + \sigma D(\gamma), \theta_t(i) > y_t(i) - \sigma D(\gamma)) \\ &\quad + \sum_{t=1}^{T} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i) + \sigma D(\gamma), \theta_t(i) < y_t(i) - \sigma D(\gamma)) \end{aligned} \quad (13)$$

The first part can be bounded by Lemma 7,

$$\sum_{t=1}^{T} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) > x_t(i) + \sigma D(\gamma)) \le T(1 - \gamma)^{48} \log \frac{1}{1 - \gamma}$$

**Step 3** The second part can be bounded through a similar method as Equation (10).

$$\begin{aligned} &\sum_{t=1}^{T} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i) + \sigma D(\gamma), \theta_t(i) > y_t(i) - \sigma D(\gamma)) \\ &\le \mathbb{E} \left[ \sum_{t=1}^{T} \mathbb{1}\{N_t(\gamma, i) > A(\gamma), \hat{\mu}_t(\gamma, i) < x_t(i) + \sigma D(\gamma)\} \mathbb{P}(\theta_t(i) > y_t(i) - \sigma D(\gamma) | \mathcal{F}_{t-1}) \right] \end{aligned} \quad (14)$$

Given the history  $\mathcal{F}_{t-1}$  such that  $N_t(\gamma, i) > A(\gamma)$  and  $\hat{\mu}_t(\gamma, i) < x_t(i) + \sigma D(\gamma)$ , we have

$$\begin{aligned} \mathbb{P}(\theta_t(i) > y_t(i) - \sigma D(\gamma) | \mathcal{F}_{t-1}) &\le \mathbb{P}(\theta_t(i) - \hat{\mu}_t(\gamma, i) > \frac{\Delta}{3} - 2\sigma D(\gamma) | \mathcal{F}_{t-1}) \\ &\le \frac{1}{2} \exp\left(-\frac{(\Delta/3 - 2\sigma D(\gamma))^2 A(\gamma)}{2}\right) \\ &\le \frac{1}{2} (1 - \gamma)^{144}. \end{aligned}$$

For other  $\mathcal{F}_{t-1}$ , the indicator term  $\mathbb{1}\{N_t(\gamma, i) > A(\gamma), \hat{\mu}_t(\gamma, i) < x_t(i) + \sigma D(\gamma)\}$  will be 0. Hence, we can bound the second part by  $\frac{T}{2}(1 - \gamma)^{144}$ .

**Step 4** Using Lemma 4, we can get

$$\begin{aligned} &\sum_{t=1}^{T} \mathbb{P}(E_t(\gamma, i), \hat{\mu}_t(\gamma, i) < x_t(i) + \sigma D(\gamma), \theta_t(i) < y_t(i) - \sigma D(\gamma)) \\ &\le CT(1 - \gamma)L(\gamma)^{-1/(1-\gamma)} \log \frac{1}{1 - \gamma}. \end{aligned}$$

Substituting all into Equation (12), we can obtain the statement of Theorem 2.

{13}------------------------------------------------

## 7. Experiments

In this section, we empirically compare the performance of DS-TS w.r.t. the state-of-the-art algorithms on Bernoulli and an arbitrarily generated bounded reward distributions. Specifically, we compare DS-TS with Thompson Sampling (TS) (Thompson, 1933) to evaluate the improvement obtained thanks to the employment of the discounted factor  $\gamma$ . We also compare DS-TS with SW-TS (Trovo et al., 2020) to evaluate the performance of sliding window and discounted factor. Furthermore, we compare DS-TS with another discounted method, DS-UCB (Garivier & Moulines, 2011), to evaluate the effect of Thompson Sampling and UCB. Moreover, we compare DS-TS with some novel and efficient algorithms such as CUSUM (Liu et al., 2018), M-UCB (Cao et al., 2019) and LB-SDA (Baudry et al., 2021). We measure the performance of each algorithm with the cumulative expected regret defined in Equation (1). The expected regret is averaged on 100 independently runs. The 95% confidence interval is obtained by performing 100 independent runs and is shown as a semi-transparent region in the figure.

### 7.1 Abruptly Changing Settings

**Experimental Setting** The time horizon is set as  $T = 100000$ . We split the time horizon into 5, 10, 20 phases of equal length and use a number of arms  $K = \{5, 10, 20, 30\}$ , respectively. We only show the results of some of the experimental settings, one can run the code on online website to get more results<sup>1</sup>.

![Figure 2: A line plot showing the expected reward mu_t(i) for five arms (Arm 1 to Arm 5) over 100,000 rounds. The x-axis is 'Round t' from 0 to 100,000. The y-axis is 'mu_t(i)' from 0.0 to 1.0. The plot shows five distinct phases separated by vertical lines at approximately 20,000, 40,000, 60,000, and 80,000 rounds. In each phase, the arms have different expected reward values, and these values change abruptly at the phase boundaries. Each arm is represented by a solid line with a semi-transparent shaded 95% confidence interval. Arm 1 is blue, Arm 2 is green, Arm 3 is red, Arm 4 is orange, and Arm 5 is purple.](411fa16c3211377525ba37c57784fee0_img.jpg)

Figure 2: A line plot showing the expected reward mu\_t(i) for five arms (Arm 1 to Arm 5) over 100,000 rounds. The x-axis is 'Round t' from 0 to 100,000. The y-axis is 'mu\_t(i)' from 0.0 to 1.0. The plot shows five distinct phases separated by vertical lines at approximately 20,000, 40,000, 60,000, and 80,000 rounds. In each phase, the arms have different expected reward values, and these values change abruptly at the phase boundaries. Each arm is represented by a solid line with a semi-transparent shaded 95% confidence interval. Arm 1 is blue, Arm 2 is green, Arm 3 is red, Arm 4 is orange, and Arm 5 is purple.

Figure 2:  $K = 5, B_T = 10$  for Bernoulli rewards.

At each breakpoint, the expected value  $\mu_t(i)$  of each arm  $i$  is drawn from a uniform distribution over  $[0, 1]$ . In the stationary phase, the rewards distributions remain unchanged. The Bernoulli arms for each phase are generated as  $\mu_t(i) \sim U(0, 1)$ . Figure 2 depicts the expected rewards for Bernoulli arms with  $K = 5$  and  $B_T = 10$ .

Based on Corollary 1, we set  $\gamma = 1 - \sqrt{B_T/T}$ .  $\tau_{max}$  is an important parameter that not only ensures the exploration ability of the algorithm but also prevents the sampling

1. Our code is available at <https://github.com/qh1874/nonmab>

{14}------------------------------------------------

deviating too much from the arm's expectation.  $\tau_{max}$  is generally  $1/5 - 1/3$  of the upper bound of the expected rewards. In this experiment, the upper bound of expected rewards  $\mu_{max} = 1$ , we take  $\tau_{max}$  as  $1/5$ . To allow for fair comparison, DS-UCB uses the discount factor  $\gamma = 1 - \sqrt{B_T/T}/4$ ,  $B = 1$ ,  $\xi = 2/3$  suggested by Garivier and Moulines (2011). Based on (Baudry et al., 2021), we set  $\tau = 2\sqrt{T \log(T)}/B_T$  for LB-SDA and SW-TS. For changepoint detection algorithm M-UCB, we set  $w = 800$ ,  $b = \sqrt{w/2 \log(2KT^2)}$  suggested by Cao et al. (2019). But set the amount of exploration  $\gamma = \sqrt{KB_T \log(T)/T}$ . In practice, it has been found that using this value instead of the one guaranteed in (Cao et al., 2019) will improve empirical performance (Baudry et al., 2021). For CUSUM, following from (Liu et al., 2018), we set  $\alpha = \sqrt{B_T/T \log(T/B_T)}$  and  $h = \log(T/B_T)$ . For our experiment settings, we choose  $M = 50$ ,  $\epsilon = 0.05$ . Based on (Auer et al., 2002), the parameters  $\alpha$  and  $\gamma$  for EXP3S are set as follows:  $\alpha = 1/T$ ,  $\gamma = \min(1, \sqrt{K(e + B_T \log(KT))/(e - 1)T})$ .

![Figure 3: Four line plots (a, b, c, d) showing Regret vs. Round t for abruptly changing settings. Each plot compares EXP3S, SW-LB-SDA, CUSUM, DS-UCB, M-UCB, TS, SW-TS, and DS-TS. The x-axis is Round t (0 to 100000) and the y-axis is Regret (0 to 20000). In all cases, DS-TS and SW-TS perform best, followed by DS-UCB and CUSUM. EXP3S, M-UCB, and TS show significantly higher regret.](eaae122ace5c0d761133c6ce971a6ffd_img.jpg)

Figure 3 consists of four subplots labeled (a), (b), (c), and (d), each showing the Regret (Y-axis) versus Round  $t$  (X-axis) for various algorithms. The algorithms compared are EXP3S, SW-LB-SDA, CUSUM, DS-UCB, M-UCB, TS, SW-TS, and DS-TS. The X-axis for all plots ranges from 0 to 100,000. The Y-axis scales vary: (a) 0 to 20,000, (b) 0 to 25,000, (c) 0 to 25,000, and (d) 0 to 35,000. In all plots, DS-TS and SW-TS consistently show the lowest regret, followed by DS-UCB and CUSUM. EXP3S, M-UCB, and TS show significantly higher regret, with TS exhibiting the highest regret in all cases. The settings for each plot are: (a)  $K=5, B_T=10$ , (b)  $K=10, B_T=10$ , (c)  $K=20, B_T=10$ , and (d)  $K=30, B_T=10$ .

Figure 3: Four line plots (a, b, c, d) showing Regret vs. Round t for abruptly changing settings. Each plot compares EXP3S, SW-LB-SDA, CUSUM, DS-UCB, M-UCB, TS, SW-TS, and DS-TS. The x-axis is Round t (0 to 100000) and the y-axis is Regret (0 to 20000). In all cases, DS-TS and SW-TS perform best, followed by DS-UCB and CUSUM. EXP3S, M-UCB, and TS show significantly higher regret.

Figure 3: Abruptly changing settings. Settings with  $K = 5, B_T = 10$  (a),  $K = 10, B_T = 10$  (b),  $K = 20, B_T = 10$  (c) and  $K = 30, B_T = 10$  (d).

**Results** Figure 3 report the results for Bernoulli arms in abruptly changing settings. It can be observed that our method and SW-TS has almost the same performance. Thompson Sampling (TS) is an algorithm for stationary MAB problems, so it oscillates a lot at the

{15}------------------------------------------------

breakpoint. The changepoint detection algorithm CUSUM also shows competitive performance. Note that, our experiment does not satisfy the detectability assumption of CUSUM (Liu et al., 2018). When the number of arms are large, several algorithms, such as EXP3S and DS-UCB, have a near-linear regret, while our algorithm still performs well. While two TS-based algorithms, DS-TS and SW-TS, still work well. This is consistent with the results in (Bayati, Hamidi, Johari, & Khosravi, 2020): when the number of arms is relatively large, algorithms based on TS are often better than that of UCB-class algorithms.

![Figure 4: Four subplots (a, b, c, d) showing Regret vs. Delta_T for different settings of K and B_T. Each plot compares DS-TS (red circles), DS-UCB (blue triangles), CUSUM (black squares), and SW-TS (yellow diamonds). In all cases, DS-TS and DS-UCB show high regret (near-linear or quadratic) that increases with Delta_T, while CUSUM and SW-TS show significantly lower regret that is less affected by Delta_T.](b05fbb6a015ea153c1e25245772b1a1b_img.jpg)

Figure 4 consists of four subplots labeled (a), (b), (c), and (d), each showing the relationship between Regret (Y-axis) and  $\Delta_T$  (X-axis) for four algorithms: DS-TS (red circles), DS-UCB (blue triangles), CUSUM (black squares), and SW-TS (yellow diamonds). The settings for each subplot are as follows:

- (a)  $K = 2, B_T = 5$ : Regret ranges from 0 to 3000. DS-TS and DS-UCB show high regret, while CUSUM and SW-TS show low regret.
- (b)  $K = 2, B_T = 10$ : Regret ranges from 0 to 3000. DS-TS and DS-UCB show high regret, while CUSUM and SW-TS show low regret.
- (c)  $K = 5, B_T = 5$ : Regret ranges from 0 to 8000. DS-TS and DS-UCB show high regret, while CUSUM and SW-TS show low regret.
- (d)  $K = 5, B_T = 10$ : Regret ranges from 0 to 8000. DS-TS and DS-UCB show high regret, while CUSUM and SW-TS show low regret.

Figure 4: Four subplots (a, b, c, d) showing Regret vs. Delta\_T for different settings of K and B\_T. Each plot compares DS-TS (red circles), DS-UCB (blue triangles), CUSUM (black squares), and SW-TS (yellow diamonds). In all cases, DS-TS and DS-UCB show high regret (near-linear or quadratic) that increases with Delta\_T, while CUSUM and SW-TS show significantly lower regret that is less affected by Delta\_T.

Figure 4:  $\Delta_T$  along with regret. Settings with  $K = 2, B_T = 5$  (a),  $K = 2, B_T = 10$  (b),  $K = 5, B_T = 5$  (c) and  $K = 5, B_T = 10$  (d).

**Impact of  $\Delta_T$**  The regret upper bound of DS-TS as well as CUSUM, DS-UCB and SW-TS, all depend on  $\Delta_T$ . In order to more clearly analyze the impact of  $\Delta_T$  on the performance, we consider the environment:  $K = \{2, 5\}, B_T = \{5, 10\}, T = 10000$ . We let  $\Delta_T$  vary within the interval  $(0, 0.3)$ , and compare the regrets of CUSUM, DS-UCB, SW-TS and DS-TS. As shown in Figure 4, the performance of CUSUM and SW-TS is less significantly affected by  $\Delta_T$ . The reason behind this phenomenon is that their regret upper bound have different factors about  $\Delta_T$ . Their regret upper bounds are shown in Table 7.1. As can be seen from the table, the upper bound of CUSUM with respect to  $\Delta_T$  is

{16}------------------------------------------------

only related to  $B_T$ . The upper bound of SW-TS has factor  $\frac{1}{\Delta_T}$ , while DS-UCB and DS-TS have factor  $\frac{1}{(\Delta_T)^2}$ . However, DS-UCB and DS-TS also have smaller regret when  $\Delta_T$  is small, as  $\Delta_T$  increases, the regret value increases first and then tends to decrease or stabilize gradually. This is because when  $\Delta_T$  is very small, although the algorithm cannot distinguish the optimal arm from the suboptimal arms well, the regret of choosing the suboptimal arm is small enough.

Table 1: Comparison of regret bounds related to  $\Delta_T$  in various algorithms

| Algorithm | CUMSUM                                              | SW-TS                                     | DS-UCB                                        | DS-TS                                         |
|-----------|-----------------------------------------------------|-------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| Bound     | $\tilde{O}(\frac{B_T}{(\Delta_T)^2} + \sqrt{TB_T})$ | $\tilde{O}(\frac{\sqrt{TB_T}}{\Delta_T})$ | $\tilde{O}(\frac{\sqrt{TB_T}}{(\Delta_T)^2})$ | $\tilde{O}(\frac{\sqrt{TB_T}}{(\Delta_T)^2})$ |

### 7.2 Smoothly Changing Settings

**Experimental Setting** We use a number of arms  $K = \{5, 10, 20, 30\}$  and the time horizon is set as  $T = \{10^4, 10^5\}$ . The smoothly changing setting we use is the same as (Trovo et al., 2020) and (Combes & Proutiere, 2014), where the expected reward changing periodically according to the following function:

$$\mu_t(i) = \frac{K-1}{K} - \frac{|w(t) - i|}{K} \\ w(t) = 1 + \frac{(K-1)(1 + \sin(t\sigma))}{2} \quad (15)$$

The expected value generated from Equation (15) clearly satisfies Assumption 1. Trovo et al. (2020) have shown that  $F = \frac{4K}{\sigma(K-1)}$ ,  $\Delta_0 = \frac{1}{3}$  satisfies Assumption 2 regardless of the value of  $\beta$ .

![Figure 5: Two plots showing expected rewards over time. Plot (a) shows K=5, T=10^4, sigma=0.001, with rewards oscillating between 0.1 and 0.8. Plot (b) shows K=5, T=10^4, sigma=0.0001, with rewards changing more gradually between 0.1 and 0.8.](250cf77a1cd51989da09fca796b3e4ea_img.jpg)

Figure 5 consists of two line plots, (a) and (b), showing the expected reward  $\mu_t(i)$  for five arms (Arm 1 to Arm 5) over 10,000 rounds. The y-axis for both plots ranges from 0.1 to 0.8, and the x-axis ranges from 0 to 10,000 rounds.

Plot (a) shows a rapidly oscillating periodic function with a period of approximately 1,000 rounds. The rewards for each arm fluctuate between approximately 0.1 and 0.8 in a regular, wave-like pattern.

Plot (b) shows a much smoother, nearly linear decreasing trend for all arms. The rewards start higher at round 0 (around 0.75 to 0.85) and decrease steadily to lower values by round 10,000 (around 0.1 to 0.2).

Figure 5: Two plots showing expected rewards over time. Plot (a) shows K=5, T=10^4, sigma=0.001, with rewards oscillating between 0.1 and 0.8. Plot (b) shows K=5, T=10^4, sigma=0.0001, with rewards changing more gradually between 0.1 and 0.8.

Figure 5: Instances of expected rewards change over the time. Settings with  $K = 5, T = 10^4, \sigma = 0.001$  (a),  $K = 5, T = 10^4, \sigma = 0.0001$  (b).

Unlike the abruptly changing settings, we do not compare the CUMSUM and M-UCB algorithms. Both algorithms use the opposite assumption to Assumption 1 to guarantee

{17}------------------------------------------------

detectability. Let  $B_T = 1$  (i.e.  $b_1 = 1$ , this means there is no breakpoint), other algorithms including ours use the same parameter as abruptly changing settings. Since we want to get a regret for  $\tilde{O}(\sqrt{T})$ , in this experiment we set  $\beta = \frac{1}{2}$ . Corollary 2 suggests taking  $\gamma = 1 - \frac{1}{\sqrt{T}}$ , in this experiment we take  $\gamma = 1 - \frac{10}{\sqrt{T}}$  to ensure that the conditions of Theorem 2 are satisfied. In particular, if  $T = 10^4$ ,  $\sigma = 0.001$  or  $T = 10^5$ ,  $\sigma = 0.0001$ ,  $\gamma = 1 - \frac{10}{\sqrt{T}}$  satisfies Theorem 2 while  $\gamma = 1 - \frac{1}{\sqrt{T}}$  not. Figure 5 shows two instances of smoothly changing arms.

![Figure 6: Four subplots (a, b, c, d) showing Regret vs. Round t for six algorithms: EXPIS (green), SW-LB-SDA (blue), DS-UCB (red), TS (orange), SW-TS (yellow), and DS-TS (purple). The x-axis represents Round t from 0 to 10000, and the y-axis represents Regret. (a) K=5, T=10^4, sigma=0.001; (b) K=5, T=10^5, sigma=0.0001; (c) K=5, T=10^4, sigma=0.0001; (d) K=10, T=10^5, sigma=0.0001. In all cases, EXPIS and SW-LB-SDA show the highest regret, while SW-TS and DS-TS show the lowest regret.](8ccbc9fa77bf60ba0ca0b79dec8681b8_img.jpg)

Figure 6: Four subplots (a, b, c, d) showing Regret vs. Round t for six algorithms: EXPIS (green), SW-LB-SDA (blue), DS-UCB (red), TS (orange), SW-TS (yellow), and DS-TS (purple). The x-axis represents Round t from 0 to 10000, and the y-axis represents Regret. (a) K=5, T=10^4, sigma=0.001; (b) K=5, T=10^5, sigma=0.0001; (c) K=5, T=10^4, sigma=0.0001; (d) K=10, T=10^5, sigma=0.0001. In all cases, EXPIS and SW-LB-SDA show the highest regret, while SW-TS and DS-TS show the lowest regret.

Figure 6: Smoothly changing settings. Settings with  $K = 5$ ,  $T = 10^4$ ,  $\sigma = 0.001$  (a),  $K = 5$ ,  $T = 10^4$ ,  $\sigma = 0.0001$  (b),  $K = 5$ ,  $T = 10^5$ ,  $\sigma = 0.0001$  (c) and  $K = 10$ ,  $T = 10^5$ ,  $\sigma = 0.0001$  (d).

## Results

Figure 6 report the results for smoothly changing settings. It can be seen that SW-TS and SW-LB-SDA achieve similar performance in several environmental settings. Due to the extra logarithmic regret induced by the discounted method while adapting to changes in the reward, the performance of DS-TS is not as potent as that of SW-TS and SW-LB-SDA. However, when  $T = 10^4$ ,  $\sigma = 0.0001$ , Thompson Sampling exhibits excellent performance.

{18}------------------------------------------------

The reason for this phenomenon can be explained by Figure 5(b). In this environment the optimal arm is switched only twice and the difference between the optimal arm and the second best arm is not significant.

### 7.3 Prior Knowledge of $\mu_{max}$

The expectation of arm in our experiment is uniformly sampled from  $(0, 1)$ . For a reasonable number of arms, at least one of the arms has a expected value close to 1. Most of the algorithms based on UCB and TS perform well due to their strong exploration ability. Now we change the experimental settings so that the expectation of the arms are relatively small and test the performance of each algorithm.

In abruptly changing settings, we limit the maximum expectation of the arms to less than 0.7. In smoothly changing settings, we limit the maximum expectation to less than 0.5. To this end, we modify the expected arms generation function (15) as  $\mu_t(i) = \frac{K}{2(K-1)}\mu_t(i)$ . We test the performance of each algorithm using the same parameter as before except DS-TS. We test DS-TS with  $\tau_{max} = 1/5$  and  $\tau_{max} = \mu_{max}/5$  respectively. The latter means DS-TS has additional information about  $\mu_{max}$ .

![Figure 7: Comparison of algorithm performance in small expected reward settings. (a) Abruptly changing settings with K=10, B_T=10, T=10^5. (b) Smoothly changing settings with K=5, T=10^4, sigma=0.001. Both plots show Regret vs. Round t for EXP3, SW-LB-SDA, CUSUM, DS-UCB, M-UCB, TS, SW-TS, DS-TS, and DS-TS(know mu_max).](3468bcffa38de23cef94bfb460ccb301_img.jpg)

Figure 7 consists of two line plots, (a) and (b), showing the regret of various algorithms over time (Round t). The algorithms compared are EXP3, SW-LB-SDA, CUSUM, DS-UCB, M-UCB, TS, SW-TS, DS-TS, and DS-TS(know  $\mu_{max}$ ).

Plot (a) shows the performance in abruptly changing settings. The x-axis is 'Round t' from 0 to 100,000, and the y-axis is 'Regret' from 0 to 20,000. The regret for all algorithms increases over time, with DS-TS(know  $\mu_{max}$ ) showing the lowest regret, followed by DS-TS. Other algorithms like EXP3 and SW-LB-SDA show significantly higher regret.

Plot (b) shows the performance in smoothly changing settings. The x-axis is 'Round t' from 0 to 10,000, and the y-axis is 'Regret' from 0 to 1,000. The regret for all algorithms increases over time. DS-TS(know  $\mu_{max}$ ) again shows the lowest regret, followed by DS-TS. The regret values are generally lower in this plot compared to plot (a), reflecting the smoother environment.

Figure 7: Comparison of algorithm performance in small expected reward settings. (a) Abruptly changing settings with K=10, B\_T=10, T=10^5. (b) Smoothly changing settings with K=5, T=10^4, sigma=0.001. Both plots show Regret vs. Round t for EXP3, SW-LB-SDA, CUSUM, DS-UCB, M-UCB, TS, SW-TS, DS-TS, and DS-TS(know mu\_max).

Figure 7: (a) Abruptly changing settings with expected value less than 0.7. Settings with  $K = 10, B_T = 10, T = 10^5$ . (b) Smoothly changing settings with expected value less than 0.5. Settings with  $K = 5, T = 10^4, \sigma = 0.001$ .

Figure 7 shows the performance of each algorithm with small expected reward. First, a comparison of Figure 3(b) shows that the regret values for each algorithm are larger than the case where the expected rewards are sampled uniformly from  $(0, 1)$ . In smoothly changing settings, the expected reward was scaled down, meaning that the reward of the arm changed more slowly. Comparison with Figure 1 reveal that the regret of each algorithm even decreased except for DS-TS. This suggests that DS-TS is more influenced by  $\mu_{max}$  than the other algorithms. Second, if the  $\mu_{max}$  is known in advance and the  $\tau_{max}$  of DS-TS is reset to  $\mu_{max}/5$ , the performance of DS-TS will be significantly improved.

{19}------------------------------------------------

## 8. Discussion

In this paper, we have proposed DS-TS algorithm with Gaussian priors for abruptly changing and smoothly changing MAB problems. Under mild assumptions, we provide the regret upper bounds of DS-TS in both non-stationary settings. Our experiments show that DS-TS can achieve significant regret reduction with respect to the state-of-the-art algorithms when the  $\mu_{max}$  is known in advance. Furthermore, we empirically analyze the influence of  $\Delta_T$  on the performance of different algorithms.

However, there are still some shortcomings in our work. First, the performance of DS-TS is usually similar to that of SW-TS in abruptly changing settings, but the regret upper bound of DS-TS has an extra logarithmic term  $\log T$ , which is probably due to the fact that our analysis method yields a too rough upper bound. Second, it’s natural to use the Bernoulli prior for bounded rewards. As  $N_t(\gamma, i)$  is no longer a positive integer in discounted method, the relationship between binomial distribution and beta distribution cannot be used for analysis. According to the literature on tail bounds of the beta distribution (Zhang & Zhou, 2020), the method in this paper cannot be used to analyze DS-TS with Bernoulli priors. Addressing these shortcomings could be a future research direction.

## References

- Abramowitz, M., & Stegun, I. A. (1964). *Handbook of mathematical functions with formulas, graphs, and mathematical tables*, Vol. 55. US Government printing office.
- Agrawal, S., & Goyal, N. (2013). Further optimal regret bounds for thompson sampling. In *Artificial intelligence and statistics*, pp. 99–107. PMLR.
- Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). The nonstochastic multiarmed bandit problem. *SIAM journal on computing*, 32(1), 48–77.
- Auer, P., Gajane, P., & Ortner, R. (2019). Adaptively tracking the best bandit arm with an unknown number of distribution changes. In *Conference on Learning Theory*, pp. 138–158. PMLR.
- Baudry, D., Russac, Y., & Cappé, O. (2021). On limited-memory subsampling strategies for bandits. In *International Conference on Machine Learning*, pp. 727–737. PMLR.
- Bayati, M., Hamidi, N., Johari, R., & Khosravi, K. (2020). Unreasonable effectiveness of greedy algorithms in multi-armed bandit with many arms. *Advances in Neural Information Processing Systems*, 33, 1713–1723.
- Besbes, O., Gur, Y., & Zeevi, A. (2014). Stochastic multi-armed-bandit problem with non-stationary rewards. *Advances in neural information processing systems*, 27.
- Besson, L., Kaufmann, E., Maillard, O.-A., & Seznec, J. (2022). Efficient change-point detection for tackling piecewise-stationary bandits. *Journal of Machine Learning Research*, 23(77), 1–40.
- Bouneffouf, D., Bouzeghoub, A., & Ganarski, A. L. (2012). A contextual-bandit algorithm for mobile context-aware recommender system. In *International conference on neural information processing*, pp. 324–331. Springer.

{20}------------------------------------------------

- Cao, Y., Wen, Z., Kveton, B., & Xie, Y. (2019). Nearly optimal adaptive procedure with change detection for piecewise-stationary bandit. In *The 22nd International Conference on Artificial Intelligence and Statistics*, pp. 418–427. PMLR.
- Chen, Y., Lee, C.-W., Luo, H., & Wei, C.-Y. (2019). A new algorithm for non-stationary contextual bandits: Efficient, optimal and parameter-free. In *Conference on Learning Theory*, pp. 696–726. PMLR.
- Combes, R., & Proutiere, A. (2014). Unimodal bandits: Regret lower bounds and optimal algorithms. In *International Conference on Machine Learning*, pp. 521–529. PMLR.
- Garivier, A., & Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. In *International Conference on Algorithmic Learning Theory*, pp. 174–188. Springer.
- Kocsis, L., & Szepesvári, C. (2006). Discounted ucb. In *2nd PASCAL Challenges Workshop*, Vol. 2, pp. 51–134.
- Li, L., Chu, W., Langford, J., & Wang, X. (2011). Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms. In *Proceedings of the fourth ACM international conference on Web search and data mining*, pp. 297–306.
- Li, S., Karatzoglou, A., & Gentile, C. (2016). Collaborative filtering bandits. In *Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval*, pp. 539–548.
- Liu, F., Lee, J., & Shroff, N. (2018). A change-detection based framework for piecewise-stationary multi-armed bandit problem. In *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Mellor, J., & Shapiro, J. (2013). Thompson sampling in switching environments with bayesian online change detection. In *Artificial intelligence and statistics*, pp. 442–450. PMLR.
- Raj, V., & Kalyani, S. (2017). Taming non-stationary bandits: A bayesian approach. *arXiv preprint arXiv:1707.09727*.
- Robbins, H. (1952). Some aspects of the sequential design of experiments. *Bulletin of the American Mathematical Society*, 58(5), 527–535.
- Schwartz, E. M., Bradlow, E. T., & Fader, P. S. (2017). Customer acquisition via display advertising using multi-armed bandit experiments. *Marketing Science*, 36(4), 500–522.
- Suk, J., & Kpotufe, S. (2022). Tracking most significant arm switches in bandits. In *Conference on Learning Theory*, pp. 2160–2182. PMLR.
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4), 285–294.
- Trovo, F., Paladino, S., Restelli, M., & Gatti, N. (2020). Sliding-window thompson sampling for non-stationary settings. *Journal of Artificial Intelligence Research*, 68, 311–364.
- Wu, Q., Iyer, N., & Wang, H. (2018). Learning contextual bandits in a non-stationary environment. In *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval*, pp. 495–504.

{21}------------------------------------------------

Zhang, A. R., & Zhou, Y. (2020). On the non-asymptotic and sharp lower tail bounds of random variables. *Stat*, 9(1), e314.

{22}------------------------------------------------

## Appendix A. Facts and Lemmas

The following inequality is the anti-concentration and concentration bound for Gaussian distributed random variables.

**Fact 1** ((Abramowitz & Stegun, 1964)). For a Gaussian distributed random variable  $X$  with mean  $\mu$  and variance  $\sigma^2$ , for any  $a > 0$

$$\frac{1}{\sqrt{2\pi}} \frac{a}{1+a^2} e^{-a^2/2} \le \mathbb{P}(X - \mu > a\sigma) \le \frac{1}{a + \sqrt{a^2 + 4}} e^{-a^2/2}$$

The following lemma is adapted from (Agrawal & Goyal, 2013) and is often used in the analysis of Thompson Sampling, can transform the probability of selecting the  $i$ th arm into the probability of selecting the optimal arm  $i_t^*$ .

**Lemma 5.** Let  $p_{i,t} = \mathbb{P}(\theta_t(*) > y_t(i)|\mathcal{F}_{t-1})$ . For any  $A > 0$ ,  $i \neq i_t^*$ ,

$$\mathbb{P}(i_t = i, \theta_t(i) < y_t(i)|\mathcal{F}_{t-1}) \le \frac{(1 - p_{i,t})}{p_{i,t}} \mathbb{P}(i_t = i_t^*, \theta_t(i) < y_t(i)|\mathcal{F}_{t-1})$$

**Lemma 6** ((Garivier & Moulines, 2011)). For any  $i \in \{1, \dots, K\}$ ,  $\gamma \in (0, 1)$  and  $A > 0$ ,

$$\sum_{t=1}^{T} \mathbb{1}\{i_t = i, N_t(\gamma, i) < A\} \le [T(1 - \gamma)]A\gamma^{-1/(1-\gamma)}.$$

**Lemma 7.** For abruptly changing settings,

$$\mathbb{P}(\hat{\mu}_t(\gamma, i) > \mu_t(i) + \frac{\Delta_t(i)}{3}, N_t(\gamma, i) > A(\gamma)) \le (1 - \gamma)^{48} \log \frac{1}{1 - \gamma}$$

For smoothly changing settings,

$$\mathbb{P}(\hat{\mu}_t(\gamma, i) > \mu_t(i) + \frac{\Delta_t(i)}{3} + \sigma D(\gamma), N_t(\gamma, i) > A(\gamma)) \le (1 - \gamma)^{48} \log \frac{1}{1 - \gamma}$$

*Proof.* Recall that,  $m = \frac{12\sqrt{2}}{\sqrt{1-\gamma}} + 3$ ,  $n = 12\sqrt{2} + 3\sqrt{1-\gamma}$ . In the abruptly changing settings,

$A(\gamma) = \frac{n^2 \log(\frac{1}{1-\gamma})}{(\Delta\gamma)^2}$ . We have

$$\begin{aligned} & \mathbb{P}(\hat{\mu}_t(\gamma, i) > \mu_t(i) + \frac{\Delta_t(i)}{3}, N_t(\gamma, i) > A(\gamma)) \\ &= \mathbb{P}(\hat{\mu}_t(\gamma, i) - \hat{\mu}_t(\gamma, i) > \mu_t(i) + \frac{\Delta_t(i)}{3} - \hat{\mu}_t(\gamma, i), N_t(\gamma, i) > A(\gamma)) \\ &\stackrel{(a)}{\le} \mathbb{P}\left(\frac{N_t(\gamma, i)(\hat{\mu}_t(\gamma, i) - \hat{\mu}_t(\gamma, i))}{\sqrt{N_t(\gamma^2, i)}} > \frac{N_t(\gamma, i)}{\sqrt{N_t(\gamma^2, i)}} \left(\frac{\Delta_T}{3} - U_t(\gamma, i)\right), N_t(\gamma, i) > A(\gamma)\right) \\ &\stackrel{(b)}{\le} \mathbb{P}\left(\frac{N_t(\gamma, i)(\hat{\mu}_t(\gamma, i) - \hat{\mu}_t(\gamma, i))}{\sqrt{N_t(\gamma^2, i)}} > \left(\frac{1}{3} - \frac{1}{m}\right)\Delta_T\sqrt{A(\gamma)}\right) \\ &\stackrel{(c)}{\le} \frac{\log \frac{1}{1-\gamma}}{\log(1+\gamma)} \exp\left(-2\left(\frac{1}{3} - \frac{1}{m}\right)\Delta_T^2 A(\gamma)\left(1 - \frac{\eta^2}{16}\right)\right) \\ &\le \frac{\log \frac{1}{1-\gamma}}{\log(1+\gamma)} \exp\left(-64 \log\left(\frac{1}{1-\gamma}\right)\left(1 - \frac{\eta^2}{16}\right)\right) \end{aligned} \tag{16}$$

{23}------------------------------------------------

where (a) uses Lemma 1, (b) follows from  $N_t(\gamma, i) > N_t(\gamma^2, i)$ ,  $\Delta_T < \Delta_t(i)$  and Equation (4), (c) uses the self-normalized Hoeffding-type inequality (Garivier & Moulines, 2011). Let  $\eta = 2$ , we can obtain the statement for abruptly changing settings.

For smoothly changing settings,

$$\begin{aligned}
& \mathbb{P}(\hat{\mu}_t(\gamma, i) > \mu_t(i) + \frac{\Delta_t(i)}{3} + \sigma D(\gamma), N_t(\gamma, i) > A(\gamma)) \\
&= \mathbb{P}(\hat{\mu}_t(\gamma, i) - \bar{\mu}_t(\gamma, i) > \mu_t(i) + \frac{\Delta_t(i)}{3} + \sigma D(\gamma) - \bar{\mu}_t(\gamma, i), N_t(\gamma, i) > A(\gamma)) \\
&\stackrel{(d)}{\le} \mathbb{P}(\frac{N_t(\gamma, i)(\hat{\mu}_t(\gamma, i) - \bar{\mu}_t(\gamma, i))}{\sqrt{N_t(\gamma^2, i)}} > \frac{N_t(\gamma, i)}{\sqrt{N_t(\gamma^2, i)}}(\frac{\Delta}{3} - U_t(\gamma, i)), N_t(\gamma, i) > A(\gamma)) \quad (17) \\
&\le \frac{\log \frac{1}{1-\gamma}}{\log(1+\eta)} \exp(-64 \log(\frac{1}{1-\gamma})(1 - \frac{\eta^2}{16}))
\end{aligned}$$

where (d) uses the Lemma 3. Let  $\eta = 2$ , we obtain the conclusion.  $\square$

## Appendix B. Proofs of Lemmas

### B.1 Proof of Lemma 1

Let  $M_t(\gamma, i) = \sum_{j=1}^t \gamma^{t-j} \mu_j(i) \mathbb{1}\{i_j = i\}$ ,  $\hat{\mu}_t(\gamma, i) = \frac{M_t(\gamma, i)}{N_t(\gamma, i)} \in [0, 1]$  is a convex combination of elements  $\mu_j(i)$ ,  $j = 1, \dots, t$ . For  $t \in \mathcal{T}(\gamma)$ ,

$$\begin{aligned}
|\mu_t(i) - \hat{\mu}_t(\gamma, i)| &= \frac{1}{N_t(\gamma, i)} |M_t(\gamma, i) - \mu_t(i) N_t(\gamma, i)| \\
&= \frac{1}{N_t(\gamma, i)} \left| \sum_{j=1}^{t-D(\gamma)} \gamma^{t-j} (\mu_j(i) - \mu_t(i)) \mathbb{1}\{i_j = i\} \right| \\
&\le \frac{1}{N_t(\gamma, i)} \sum_{j=1}^{t-D(\gamma)} \gamma^{t-j} \mathbb{1}\{i_j = i\} \\
&= \frac{1}{N_t(\gamma, i)} \gamma^{D(\gamma)} N_{t-D(\gamma)}(\gamma, i) \\
&\stackrel{(a)}{\le} \frac{\gamma^{D(\gamma)}}{N_t(\gamma, i)(1-\gamma)} \\
&\stackrel{(b)}{\le} \sqrt{\frac{\gamma^{D(\gamma)}}{N_t(\gamma, i)(1-\gamma)}},
\end{aligned}$$

where (a) follows from  $N_{t-D(\gamma)}(\gamma, i) \le \frac{1}{1-\gamma}$ , (b) follows from  $|\mu_t(i) - \hat{\mu}_t(\gamma, i)| \le 1$  and  $1 \wedge x \le \sqrt{x}$ . By the definition of  $D(\gamma)$ ,

$$|\mu_t(i) - \hat{\mu}_t(\gamma, i)| \le \sqrt{\frac{-(1-\gamma) \log(1-\gamma)}{N_t(\gamma, i)}}$$

{24}------------------------------------------------

### B.2 Proof of Lemma 2

Recall that  $p_{i,t} = \mathbb{P}(\theta_t(*) > y_t(i) | \mathcal{F}_{t-1})$ ,  $A(\gamma) = \frac{n^2 \log(\frac{1}{1-\gamma})}{(\Delta\tau)^2}$ ,  $U_t(\gamma, i) = \sqrt{\frac{(1-\gamma) \log \frac{1}{1-\gamma}}{N_t(\gamma, i)}}$ ,  $m = \frac{12\sqrt{2}}{\sqrt{1-\gamma}} + 3$ ,  $n = 12\sqrt{2} + 3\sqrt{1-\gamma}$ , function  $F(x) = \frac{1}{\sqrt{2\pi}} \frac{x}{1+x^2} e^{-x^2/2}$ .

Our algorithm uses Gaussian priors  $\theta_t(i) \sim \mathcal{N}(\hat{\mu}_t(i), \min\{\frac{1}{N_t(\gamma, i)}, \tau_{\max}^2\})$ .

If  $N_t(\gamma, i) < \frac{1}{\tau_{\max}^2}$ , then  $\theta_t(i)$  is sampling from  $\mathcal{N}(\hat{\mu}_t(i), \tau_{\max}^2)$ . We have

$$p_{i,t} = \mathbb{P}(\theta_t(*) > y_t(i) | \mathcal{F}_{t-1}) \ge \mathbb{P}(\theta_t(*) - \hat{\mu}_t(*) > \mu_t(*) | \mathcal{F}_{t-1}) \ge \mathbb{P}(\theta_t(*) - \hat{\mu}_t(*) > \mu_{\max} | \mathcal{F}_{t-1})$$

Using Fact 1,  $p_{i,t} > \frac{1}{\sqrt{2\pi}} \frac{\mu_{\max}/\tau_{\max}}{1+(\mu_{\max}/\tau_{\max})^2} e^{-(\mu_{\max}/\tau_{\max})^2/2} = F(\frac{\mu_{\max}}{\tau_{\max}})$ . Note that  $\tau_{\max} > \frac{1}{12\sqrt{2}}$ , then  $N_t(\gamma, *) < \frac{1}{\tau_{\max}^2} \le A(\gamma)$ . Therefore,

$$\begin{aligned} \sum_{t \in \mathcal{T}(\gamma)} \mathbb{E}[\frac{1 - p_{i,t}}{p_{i,t}} \mathbb{1}\{i_t = i_t^*, \theta_t(i) < y_t(i)\}] &\le \sum_{t \in \mathcal{T}(\gamma)} \frac{1}{F(\frac{\mu_{\max}}{\tau_{\max}})} \mathbb{E}[\mathbb{1}\{i_t = i_t^*, N_t(\gamma, *) < A(\gamma)\}] \\ &\le \frac{1}{F(\frac{\mu_{\max}}{\tau_{\max}})} T(1-\gamma) A(\gamma) \gamma^{-1/(1-\gamma)} \end{aligned}$$

In the subsequent analyses, we can assume that  $N_t(\gamma, i) > \frac{1}{\tau_{\max}^2}$ , i.e.  $\theta_t(i) \sim \mathcal{N}(\hat{\mu}_t(i), \frac{1}{N_t(\gamma, i)})$ . The subsequent proof is in 3 steps.

**Step 1** We first prove that  $\mathbb{E}[\frac{1}{p_{i,t}}]$  has an upper bound independent of  $t$ .

Define a Bernoulli experiment as sampling from  $\mathcal{N}(\hat{\mu}_t(i), \frac{1}{N_t(\gamma, i)})$ , where success implies that  $\theta_t(i) > y_t(i)$ . Let  $G_t$  denote the number of experiments performed when the event  $\{\theta_t(i) > y_t(i)\}$  first occurs. Then

$$\mathbb{E}[\frac{1}{p_{i,t}}] = \mathbb{E}[\mathbb{E}[G_t | \mathcal{F}_{t-1}]] = \mathbb{E}[G_t]$$

Let  $z = \sqrt{\log r} + 1$  ( $r \ge 1$  is an integer) and let  $\text{MAX}_r$  denote the maximum of  $r$  independent Bernoulli experiment. Then

$$\begin{aligned} \mathbb{P}(G_t \le r) &\ge \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i)) \\ &= \mathbb{E}[\mathbb{E}[\mathbb{1}\{\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i)\} | \mathcal{F}_{t-1}]] \\ &= \mathbb{E}[\mathbb{1}\{\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i)\} \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} | \mathcal{F}_{t-1})] \end{aligned} \quad (18)$$

Using Fact 1,

$$\begin{aligned} \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} | \mathcal{F}_{t-1}) &\ge 1 - (1 - \frac{1}{\sqrt{2\pi}} \frac{z}{z^2 + 1} e^{-z^2/2})^r \\ &= 1 - (1 - \frac{1}{\sqrt{2\pi}} \frac{\sqrt{\log r} + 1}{(\sqrt{\log r} + 1)^2 + 1} e^{-1/2 - \sqrt{\log r}})^r \quad (19) \\ &\ge 1 - e^{-\frac{\sqrt{r} - \sqrt{\log r}}{\sqrt{2\pi}(\sqrt{\log r} + 2)}} \end{aligned}$$

{25}------------------------------------------------

For any  $r \ge e^{25}$ ,  $e^{-\frac{\sqrt{r}e - \sqrt{\log r}}{\sqrt{2\pi}(\sqrt{\log r} + 2)}} \le \frac{1}{r^2}$ . Hence, for any  $r \ge e^{25}$ ,

$$\mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} | \mathcal{F}_{t-1}) \ge 1 - \frac{1}{r^2}.$$

Therefore, for any  $r \ge e^{25}$ ,

$$\mathbb{P}(G_t \le r) \ge (1 - \frac{1}{r^2})\mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i))$$

Next, we apply self-normalized Hoeffding-type inequality (Garivier & Moulines, 2011) to lower bound  $\mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i))$ .

$$\begin{aligned} \mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i)) &\ge 1 - \mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \le \mu_t(*)) \\ &\ge 1 - \mathbb{P}(\hat{\mu}_t(*) - \bar{\mu}_t(*) \le U_t(\gamma, i) - \frac{z}{\sqrt{N_t(\gamma, i)}}) \\ &\stackrel{(a)}{\ge} 1 - \mathbb{P}(\hat{\mu}_t(*) - \bar{\mu}_t(*) < -\frac{\sqrt{\log r}}{\sqrt{N_t(\gamma, i)}}) \\ &\ge 1 - \frac{\log \frac{1}{1-\gamma}}{\log(1+\eta)} e^{-2 \log r (1 - \frac{\eta^2}{16})} \end{aligned}$$

where (a) follows from the fact that  $U_t(\gamma, i) - \frac{z}{\sqrt{N_t(\gamma, i)}} = \frac{\sqrt{(1-\gamma) \log \frac{1}{1-\gamma} - 1 - \sqrt{\log r}}}{\sqrt{N_t(\gamma, i)}} < -\frac{\sqrt{\log r}}{\sqrt{N_t(\gamma, i)}}$ . Let  $\eta = 2$ , we get

$$\mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i)) \ge 1 - \log \frac{1}{1-\gamma} \frac{1}{r^{1.5}}.$$

Substituting, for any  $r > e^{25}$ ,

$$\mathbb{P}(G_t \le r) \ge 1 - \log \frac{1}{1-\gamma} \frac{1}{r^{1.5}} - \frac{1}{r^2} \quad (20)$$

Therefore,

$$\begin{aligned} \mathbb{E}[G_t] &= \sum_{r=0}^{\infty} \mathbb{P}(G_t \le r) \\ &\le 1 + e^{25} + \sum_{r > e^{25}} \left( \log \frac{1}{1-\gamma} \frac{1}{r^{1.5}} + \frac{1}{r^2} \right) \\ &\le e^{25} + 3 + 3 \log \frac{1}{1-\gamma} \end{aligned}$$

This proves a bound of  $\mathbb{E}[\frac{1}{p_{i,t}}] \le e^{25} + 3 + 3 \log \frac{1}{1-\gamma}$  independent of  $t$ .

**Step 2.** Define  $L(\gamma) = \frac{144(1+\sqrt{2})^2 \log(\frac{1}{1-\gamma} + e^{25})}{\Delta_{\mathcal{I}}^2}$ . We consider the upper bound of  $\mathbb{E}[\frac{1}{p_{i,t}}]$  when  $N_t(\gamma, i) > L(\gamma)$ .

{26}------------------------------------------------

$$\begin{aligned}
\mathbb{P}(G_t \le r) &\ge \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \ge y_t(i)) \\
&= \mathbb{E}[\mathbb{1}\{\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \ge y_t(i)\} \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} | \mathcal{F}_{t-1})]
\end{aligned} \tag{21}$$

Now, since  $N_t(\gamma, i) > L(\gamma)$ ,  $\frac{1}{\sqrt{N_t(\gamma, i)}} < \frac{\Delta_t(i)}{12(1+\sqrt{2})\sqrt{\log(\frac{1}{1-\gamma} + e^{25})}}$ . Therefore, for any  $r \le (\frac{1}{1-\gamma} + e^{25})^2$ ,

$$\frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} = \frac{\sqrt{\log r} + 1}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \le -\frac{\Delta_t(i)}{12}.$$

Using Fact 1,

$$\mathbb{P}(\theta_t(i) > \hat{\mu}_t(i) - \frac{\Delta_t(i)}{12} | \mathcal{F}_{t-1}) \le 1 - \frac{1}{2} e^{-N_t(\gamma, i) \frac{\Delta_t(i)^2}{288}} \ge 1 - \frac{1}{2(1/(1-\gamma) + e^{25})^2}.$$

This implies

$$\mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} | \mathcal{F}_{t-1}) \ge 1 - \frac{1}{2r(1/(1-\gamma) + e^{25})^{2r}}.$$

Also, apply the fact that  $\frac{1}{\sqrt{N_t(\gamma, i)}} < \frac{\Delta_t(i)}{24}$  and the self-normalized Hoeffding-type inequality,

$$\begin{aligned}
\mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \ge y_t(i)) &\ge \mathbb{P}(\hat{\mu}_t(*) \ge \mu_t(*) - \frac{\Delta_t(i)}{6}) \\
&\ge 1 - \log \frac{1}{1-\gamma} \frac{1}{(1/(1-\gamma) + e^{25})^6}.
\end{aligned}$$

Let  $\gamma' = (\frac{1}{1-\gamma} + e^{25})^2$ . Therefore, for any  $1 \le r \le \gamma'$ ,

$$\mathbb{P}(G_t \le r) \ge 1 - \frac{1}{2r\gamma'^{2r}} - \log \frac{1}{1-\gamma} \frac{1}{\gamma'^6}.$$

When  $r \ge \gamma' > e^{25}$ , we can use Equation (20) to obtain,

$$\mathbb{P}(G_t \le r) \ge 1 - \log \frac{1}{1-\gamma} \frac{1}{r^{1.5}} - \frac{1}{r^2}$$

{27}------------------------------------------------

Combining these results,

$$\begin{aligned}
 \mathbb{E}[G_t] &\le \sum_{r=0}^{\infty} \mathbb{P}(G_t \ge r) \\
 &\le 1 + \sum_{r=1}^{\gamma'} \mathbb{P}(G_t \ge r) + \sum_{r=\gamma'}^{\infty} \mathbb{P}(G_t \ge r) \\
 &\le 1 + \sum_{r=1}^{\gamma'} \left( \frac{1}{2r\gamma'^{2r}} + \log \frac{1}{1-\gamma} \frac{1}{\gamma^6} \right) + \sum_{r=\gamma'}^{\infty} \left( \log \frac{1}{1-\gamma} \frac{1}{r^{1.5}} + \frac{1}{r^2} \right) \\
 &\le 1 + \frac{1}{\gamma'^2} + \log \frac{1}{1-\gamma} \frac{1}{\gamma^5} + \frac{2}{\gamma'} + \log \left( \frac{1}{1-\gamma} \frac{3}{\sqrt{\gamma'}} \right) \\
 &\le 1 + 6(1-\gamma) \log \frac{1}{1-\gamma}.
 \end{aligned}$$

Therefore, when  $N_t(\gamma, i) > L(\gamma)$ , it holds that

$$\mathbb{E}\left[\frac{1}{P_{i,t}}\right] - 1 = \mathbb{E}[G_t] - 1 \le 6(1-\gamma) \log \frac{1}{1-\gamma}.$$

**Step 3** Let  $\mathcal{A}(\gamma, i) = \{t \in \{1, \dots, T\} : i_t = i_t^*, N_t(\gamma, i) \le L(\gamma)\}$  and  $C = e^{25} + \frac{1}{F(\frac{L_{\max}}{\tau_{\max}})} + 12$ . Combined with the case where  $N_t(\gamma, i) < \frac{1}{\tau_{\max}^2}$ ,

$$\begin{aligned}
 &\sum_{t \in \mathcal{T}(\gamma)} \mathbb{E}\left[\frac{1-P_{i,t}}{P_{i,t}} \mathbb{1}\{i_t = i_t^*, \theta_t(i) < y_t(i)\}\right] \\
 &\le \sum_{t \in \mathcal{T}(\gamma) \cap \mathcal{A}(\gamma, i)} \mathbb{E}\left[\frac{1-P_{i,t}}{P_{i,t}} \mathbb{1}\{i_t = i_t^*, \theta_t(i) < y_t(i)\}\right] + \sum_{t \in \mathcal{T}(\gamma) \setminus \mathcal{A}(\gamma, i)} \mathbb{E}\left[\frac{1-P_{i,t}}{P_{i,t}} \mathbb{1}\{i_t = i_t^*, \theta_t(i) < y_t(i)\}\right] \\
 &\le |\mathcal{T}(\gamma) \cap \mathcal{A}(\gamma, i)| (e^{25} + 3 + 3 \log \frac{1}{1-\gamma}) + \frac{1}{F(\frac{L_{\max}}{\tau_{\max}})} T(1-\gamma) A(\gamma) \gamma^{-1/(1-\gamma)} + \sum_{t \in \mathcal{T}(\gamma) \setminus \mathcal{A}(\gamma, i)} \mathbb{E}\left[\frac{1-P_{i,t}}{P_{i,t}}\right] \\
 &\le T(1-\gamma) L(\gamma) \gamma^{-1/(1-\gamma)} (e^{25} + \frac{1}{F(\frac{L_{\max}}{\tau_{\max}})} + 3 + 3 \log \frac{1}{1-\gamma}) + 6T(1-\gamma) \log \frac{1}{1-\gamma} \\
 &\le CT(1-\gamma) L(\gamma) \gamma^{-1/(1-\gamma)} \log \frac{1}{1-\gamma}.
 \end{aligned} \tag{22}$$

{28}------------------------------------------------

### B.3 Proofs of Lemma 3

Let  $M_t(\gamma, i) = \sum_{j=1}^t \gamma^{t-j} \mu_j(i) \mathbb{1}\{i_j = i\}$ ,  $\tilde{\mu}_t(\gamma, i) = \frac{M_t(\gamma, i)}{N_t(\gamma, i)} \in [0, 1]$  is a convex combination of elements  $\mu_j(i)$ ,  $j = 1, \dots, t$ .

$$\begin{aligned}
 & |\mu_t(i) - \tilde{\mu}_t(\gamma, i)| \\
 &= \frac{1}{N_t(\gamma, i)} \left| \sum_{j=1}^t \gamma^{t-j} (\mu_j(i) - \mu_t(i)) \mathbb{1}\{i_j = i\} \right| \\
 &= \frac{1}{N_t(\gamma, i)} \left| \sum_{j=1}^{t-D(\gamma)} \gamma^{t-j} (\mu_j(i) - \mu_t(i)) \mathbb{1}\{i_j = i\} \right| + \frac{1}{N_t(\gamma, i)} \left| \sum_{j=t-D(\gamma)+1}^t \gamma^{t-j} (\mu_j(i) - \mu_t(i)) \mathbb{1}\{i_j = i\} \right|
 \end{aligned}$$

The first part can be bounded by  $U_t(\gamma, i)$ . Recall that the Assumption 1: There exists  $\sigma > 0$ , for all  $t, t' \ge 1, 1 \le i \le K$ , it holds that  $|\mu_t(i) - \mu_{t'}(i)| \le \sigma |t - t'|$ . Therefore,

$$\frac{1}{N_t(\gamma, i)} \left| \sum_{j=t-D(\gamma)}^t \gamma^{t-j} (\mu_j(i) - \mu_t(i)) \mathbb{1}\{i_j = i\} \right| \le \frac{\sigma D(\gamma)}{N_t(\gamma, i)} \left| \sum_{j=t-D(\gamma)}^t \gamma^{t-j} \mathbb{1}\{i_j = i\} \right| \le \sigma D(\gamma).$$

Hence, we get  $|\mu_t(i) - \tilde{\mu}_t(\gamma, i)| \le U_t(\gamma, i) + \sigma D(\gamma)$ .

### B.4 Proofs of Lemma 4

The proof of Lemma 4 is almost the same as Lemma 2. Most of the results can be obtained directly from the proof of Lemma 2, and we will only present the different parts. In smoothly changing settings,  $A(\gamma) = \frac{n^2 \log(\frac{1}{1-\gamma})}{(\Delta/3 - 2\sigma D(\gamma))^2}$ . We first bound  $\mathbb{P}(G_t \le r)$ .

$$\begin{aligned}
 \mathbb{P}(G_t \le r) &\ge \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i) - \sigma D(\gamma)) \\
 &= \mathbb{E}[\mathbb{1}\{\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i) - \sigma D(\gamma)\} \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} | \mathcal{F}_{t-1})]
 \end{aligned} \tag{23}$$

For any  $r \ge e^{25}$ ,

$$\mathbb{P}(G_t \le r) \ge \left(1 - \frac{1}{r^2}\right) \mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i) - \sigma D(\gamma))$$

We use self-normalized Hoeffding-type inequality to lower bound  $\mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i) - \sigma D(\gamma))$ .

$$\begin{aligned}
 \mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i) - \sigma D(\gamma)) &\ge 1 - \mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \le \mu_t(*) - \sigma D(\gamma)) \\
 &\stackrel{(a)}{\ge} 1 - \mathbb{P}(\hat{\mu}_t(*) - \tilde{\mu}_t(*) \le U_t(\gamma, i) - \frac{z}{\sqrt{N_t(\gamma, i)}}) \\
 &\ge 1 - \mathbb{P}(\hat{\mu}_t(*) - \tilde{\mu}_t(*) < -\frac{\sqrt{\log r}}{\sqrt{N_t(\gamma, i)}}) \\
 &\ge 1 - \frac{\log \frac{1}{1-\gamma}}{\log(1+\eta)} e^{-2\log r(1-\frac{\eta^2}{16})}
 \end{aligned}$$

{29}------------------------------------------------

where (a) follows from the fact that  $\mu_t(*) - \hat{\mu}_t(*) \le U_t(\gamma, i) + \sigma D(\gamma)$ . Let  $\eta = 2$ , we get

$$\mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} \ge y_t(i) - \sigma D(\gamma)) \ge 1 - \log \frac{1}{1-\gamma} \frac{1}{r^{1.5}}.$$

Therefore,

$$\mathbb{E}[G_t] = \sum_{r=0}^{\infty} \mathbb{P}(G_t \le r) \le e^{25} + 3 + 3 \log \frac{1}{1-\gamma}$$

Next, let  $L(\gamma) = \frac{144(1+\sqrt{2})^2 \log(\frac{1}{1-\gamma} + e^{25})}{\Delta^2}$ . We derive a tighter bound for  $N_t(\gamma, i) > L(\gamma)$ .

$$\mathbb{P}(G_t \le r)$$

$$\begin{aligned} &\ge \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \ge y_t(i) - \sigma D(\gamma)) \\ &= \mathbb{E}[\mathbb{1}\{\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \ge y_t(i) - \sigma D(\gamma)\} \mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} | \mathcal{F}_{t-1})] \end{aligned} \quad (24)$$

Since  $N_t(\gamma, i) > L(\gamma)$ ,  $\frac{1}{\sqrt{N_t(\gamma, i)}} < \frac{\Delta}{12(1+\sqrt{2})\sqrt{\log(\frac{1}{1-\gamma} + e^{25})}}$ . Therefore, for any  $r \le (\frac{1}{1-\gamma} + e^{25})^2$ ,

$$\frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \le -\frac{\Delta_t(i)}{12}.$$

Using Fact 1,

$$\mathbb{P}(\theta_t(i) > \hat{\mu}_t(i) - \frac{\Delta_t(i)}{12} | \mathcal{F}_{t-1}) \le 1 - \frac{1}{2} e^{-N_t(\gamma, i) \frac{\Delta_t(i)^2}{288}} \ge 1 - \frac{1}{2(1/(1-\gamma) + e^{25})^2}.$$

This implies

$$\mathbb{P}(\text{MAX}_r > \hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} | \mathcal{F}_{t-1}) \ge 1 - \frac{1}{2^r(1/(1-\gamma) + e^{25})^{2r}}.$$

Also, apply the fact that  $\frac{1}{\sqrt{N_t(\gamma, i)}} < \frac{\Delta}{24}$  and the self-normalized Hoeffding-type inequality,

$$\begin{aligned} &\mathbb{P}(\hat{\mu}_t(*) + \frac{z}{\sqrt{N_t(\gamma, i)}} - \frac{\Delta_t(i)}{6} \ge y_t(i) - \sigma D(\gamma)) \\ &\ge \mathbb{P}(\hat{\mu}_t(*) \ge \mu_t(*) - \sigma D(\gamma) - \frac{\Delta_t(i)}{6}) \\ &= 1 - \mathbb{P}(\hat{\mu}_t(*) - \bar{\mu}_t(*) \le \mu_t(*) - \bar{\mu}_t(*) - \sigma D(\gamma) - \frac{\Delta_t(i)}{6}) \\ &\ge 1 - \mathbb{P}(\hat{\mu}_t(*) - \bar{\mu}_t(*) \le -\frac{\Delta_t(i)}{8}) \\ &\ge 1 - \log \frac{1}{1-\gamma} \frac{1}{(1/(1-\gamma) + e^{25})^6}. \end{aligned}$$

{30}------------------------------------------------

Let  $\gamma' = (\frac{1}{1-\gamma} + e^{25})^2$ . Combining these results,

$$\begin{aligned}\mathbb{E}[G_t] &\le \sum_{r=0}^{\infty} \mathbb{P}(G_t \ge r) \\ &\le 1 + \sum_{r=1}^{\gamma'} \mathbb{P}(G_t \ge r) + \sum_{r=\gamma'}^{\infty} \mathbb{P}(G_t \ge r) \\ &\le 1 + 6(1-\gamma) \log \frac{1}{1-\gamma}.\end{aligned}$$

Therefore, when  $N_t(\gamma, i) > L(\gamma)$ , it holds that

$$\mathbb{E}\left[\frac{1}{p_{i,t}}\right] - 1 = \mathbb{E}[G_t] - 1 \le 6(1-\gamma) \log \frac{1}{1-\gamma}.$$

Follows from Equation (22) (step 3 in the proof of Lemma 2), let  $C = e^{25} + \frac{1}{F(\frac{1}{\gamma_{max}})} + 12$ , we can get

$$\sum_{t=1}^{T} \mathbb{E}\left[\frac{1 - p_{i,t}}{p_{i,t}} \mathbb{1}\{i_t = i_t^*, \theta_t(i) < y_t(i)\}\right] \le CT(1-\gamma)L(\gamma)\gamma^{-1/(1-\gamma)} \log \frac{1}{1-\gamma}.$$
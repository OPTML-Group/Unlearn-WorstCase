import  torch

def bisection(a, eps, xi=1e-5, ub=1, max_iter=1e2):
    mu_l = torch.min(a - 1)
    mu_u = torch.max(a)
    iter_count = 0
    mu_a = (mu_u + mu_l) / 2  

    while torch.abs(mu_u - mu_l) > xi:
        # print(torch.abs(mu_u - mu_l))
        mu_a = (mu_u + mu_l) / 2
        gu = torch.sum(torch.clamp(a - mu_a, 0, ub)) - eps
        gu_l = torch.sum(torch.clamp(a - mu_l, 0, ub)) - eps

        if gu == 0 or iter_count >= max_iter:
            break
        if torch.sign(gu) == torch.sign(gu_l):
            mu_l = mu_a
        else:
            mu_u = mu_a

        iter_count += 1

    upper_S_update = torch.clamp(a - mu_a, 0, ub)

    return upper_S_update


# def bisection(a, eps, xi=1e-5, ub=1, max_iter=1e2):
#     with torch.no_grad():
#         def value(a, x):
#             return torch.sum(torch.clamp(a - x, 0, ub)) - eps
#         lef = torch.min(a) - ub
#         sign = torch.sign(value(a, lef))
#         rig = torch.max(a)
        
#         for _ in range(int(max_iter)):
#             mid = (lef + rig) / 2
#             vm = value(a, mid)
#             if torch.abs(vm) < xi:
#                 break
#             if torch.sign(vm) == sign:
#                 lef = mid
#             else:
#                 rig = mid

#         result = torch.clamp(a - mid, 0, ub)

#     return result
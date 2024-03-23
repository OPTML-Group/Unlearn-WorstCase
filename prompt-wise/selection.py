import torch


def bisection(a, eps, xi=1e-5, ub=1, max_iter=1e2):
    with torch.no_grad():

        def value(a, x):
            return torch.sum(torch.clamp(a - x, 0, ub)) - eps

        lef = torch.min(a - 1)
        sign = torch.sign(value(a, lef))
        rig = torch.max(a)

        for _ in range(int(max_iter)):
            mid = (lef + rig) / 2
            vm = value(a, mid)
            if torch.abs(vm) < xi:
                break
            if torch.sign(vm) == sign:
                lef = mid
            else:
                rig = mid

        result = torch.clamp(a - mid, 0, ub)

    return result


def optimize_select(train_full_loader, model, criterion, args, w, class_wise=False):
    with torch.no_grad():
        print("################# Optimize Select #################")
        w_grad_tensor = torch.zeros(len(w)).cuda()
        for i, (image, target, index) in enumerate(train_full_loader):
            image = image.cuda()
            target = target.cuda()
            w_grad = criterion(model(image), target)
            if class_wise:
                w_grad_tensor[target] += w_grad.detach()
            else:
                w_grad_tensor[index] = w_grad.detach()

        w -= args.w_lr * (
            torch.tensor(w_grad_tensor, dtype=torch.float64).cuda() + args.gamma * 2 * w
        )
        w = bisection(w, args.num_indexes_to_replace)

        loss = torch.sum(w * w_grad_tensor)
        return w, loss

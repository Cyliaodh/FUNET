def train_model(n_epochs, model, trainLoader, valLoader, criterion, optimizer, device):
    train_cost = []
    val_cost = []
    # ouffff
    for epoch in range(n_epochs):
        cost = 0
        for batch, (cine, cine_gt, de, de_gt) in enumerate(trainLoader):
            x1, y1 = cine.to(device), cine_gt.to(device)
            x2, y2 = de.to(device), de_gt.to(device)
            pred1, pred2 = model(x1, x2)
            loss1 = criterion(pred1, y1)
            loss2 = criterion(pred2, y2)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cost += loss.data

        train_cost.append(cost)

        for c, c_gt, d, d_gt in valLoader:
            x1, y1, x2, y2 = c.to(device), c_gt.to(device), d.to(device), d_gt.to(device)

            res1, res2 = model(x1, x2)
            l1 = criterion(res1, y1)
            l2 = criterion(res2, y2)

        val_cost.append(l1 + l2)

    return train_cost, val_cost

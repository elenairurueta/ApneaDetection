from Imports import *

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    all_y_true = []
    all_y_pred = []
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        all_y_true.append(y_batch)
        all_y_pred.append(y_pred)
    avg_loss = running_loss / len(train_loader)
    y_true = torch.cat(all_y_true).detach().numpy().tolist()
    y_pred = torch.cat(all_y_pred).round().detach().numpy().tolist()
    train_acc = accuracy_score(y_true, y_pred)
    return avg_loss, train_acc

def val_one_epoch(model, val_loader, loss_fn):
    model.eval()
    val_running_loss = 0.0
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_pred = model(X_val)
            val_loss = loss_fn(y_pred, y_val)
            val_running_loss += val_loss.item()
            all_y_true.append(y_val)
            all_y_pred.append(y_pred)
    avg_val_loss = val_running_loss / len(val_loader)
    y_true = torch.cat(all_y_true)
    y_pred = torch.cat(all_y_pred).round()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return avg_val_loss, acc, f1

def Train(model, nombre, trainset, valset, n_epochs, text=""):
    start_time_training = datetime.now()

    batch_size = 32
    train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(valset, shuffle=False, batch_size=batch_size)

    loss_fn = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # Stochastic Gradient Descent
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=0.001)

    text += f'\n\nBatch size: {batch_size}\n\nLoss function: {loss_fn}\n\nOptimizer: {optimizer}'
    val_accuracies = []
    train_accuracies = []
    f1score = []
    train_losses = []
    val_losses = []
    text += '\n\nStart of training'

    for epoch in range(n_epochs):
        start_time_epoch = datetime.now()

        # Training
        avg_train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation
        avg_val_loss, val_acc, f1 = val_one_epoch(model, val_loader, loss_fn)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        f1score.append(f1)

        # Scheduler step
        #scheduler.step(avg_val_loss)

        end_of_epoch = (f"End of epoch {epoch + 1} - Accuracy = {val_acc * 100:.2f}% - F1 = {f1 * 100:.2f}% "
                        f"- Train Loss = {avg_train_loss*100:.2f}% - Val Loss = {avg_val_loss*100:.2f}% - "
                        f"{(datetime.now()-start_time_epoch).total_seconds():.2f} seconds")
                        #f" - Current lr = {scheduler.optimizer.param_groups[0]['lr']}")
        print(end_of_epoch)
        text += '\n\t' + end_of_epoch

    end_of_training = (f"End of training - {n_epochs} epochs - "
                       f"{(datetime.now()-start_time_training).total_seconds():.2f} seconds")
    print(end_of_training)
    text += '\n' + end_of_training
    plot_accuracies_losses(train_accuracies, val_accuracies, val_losses, train_losses, f1score, n_epochs, nombre)
    write_txt(nombre, text)
    return model

def plot_accuracies_losses(train_accuracies, val_accuracies, val_losses, train_losses, f1score, n_epochs, nombre):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, n_epochs + 1), train_accuracies, marker='o', color='blue', label='Train')
    plt.plot(range(1, n_epochs + 1), val_accuracies, marker='o', color='green', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.subplot(1, 3, 2)
    plt.plot(range(1, n_epochs + 1), f1score, marker='o', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Epoch vs F1 Score')
    plt.subplot(1, 3, 3)
    plt.plot(range(1, n_epochs + 1), val_losses, marker='o', color='orange', label='Val')
    plt.plot(range(1, n_epochs + 1), train_losses, marker='o', color='red', label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.tight_layout()
    
    PATH = 'models/' + nombre + '_acc_loss.png'
    plt.savefig(PATH)
    plt.show()  

def write_txt(nombre, text):
    PATH = 'models/' + nombre + '_training.txt'
    f = open(PATH, "w")
    f.write(text)
    f.close()
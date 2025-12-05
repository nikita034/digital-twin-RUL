# src/train.py
total, loss_sum = 0, 0.0
preds, trues = [], []
with torch.no_grad():
for X, y in val_loader:
X = X.to(device)
y = y.to(device)
pred = model(X)
loss = criterion(pred, y)
loss_sum += loss.item() * X.size(0)
preds.append(pred.cpu().numpy())
trues.append(y.cpu().numpy())
total += X.size(0)
preds = np.concatenate(preds)
trues = np.concatenate(trues)
rmse = np.sqrt(np.mean((preds - trues) ** 2))
mae = np.mean(np.abs(preds - trues))
return loss_sum / total, rmse, mae




def main(args):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = joblib.load(args.processed)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
feature_dim = X_train.shape[2]


train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False)


model = TransformerRegressor(input_dim=feature_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers).to(device)
opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
criterion = nn.MSELoss()


writer = SummaryWriter(log_dir=args.logdir)
best_rmse = float('inf')


os.makedirs(args.ckpt_dir, exist_ok=True)
for epoch in range(1, args.epochs + 1):
train_loss = train_loop(model, opt, criterion, train_loader, device)
val_loss, val_rmse, val_mae = eval_loop(model, criterion, val_loader, device)
print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_rmse={val_rmse:.4f} val_mae={val_mae:.4f}")
writer.add_scalar('train/loss', train_loss, epoch)
writer.add_scalar('val/rmse', val_rmse, epoch)
writer.add_scalar('val/mae', val_mae, epoch)


if val_rmse < best_rmse:
best_rmse = val_rmse
torch.save({'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'epoch': epoch}, os.path.join(args.ckpt_dir, 'best.pth'))
writer.close()




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--processed', type=str, required=True)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--logdir', type=str, default='experiments/logs')
parser.add_argument('--ckpt_dir', type=str, default='experiments/checkpoints')
args = parser.parse_args()
main(args)

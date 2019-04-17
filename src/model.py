import os
import torch
import torchvision.transforms.functional as TF


class LeNet5(torch.nn.Module):

    def __init__(self, h, w, inp, out):
        super(LeNet5, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(inp, 6, 5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(6, 16, 5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2)
        )
        self.f = 16 * (h//4 - 3) * (w//4 - 3)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.f, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, out)
        )

    def forward(self, input):
        return self.fc(self.conv(input).view(-1, self.f))


class Model(object):

    def __init__(self, use_gpu=False):
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.net = LeNet5(28, 28, 1, 10).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=0.01,
            momentum=0.8
        )

    def supervised_loss_func(self, out, gt):
        return torch.mean((out-gt)**2)

    def unsupervised_loss_func(self, out):
        fake_gt = torch.zeros(*out.shape, device=self.device)
        indices = torch.argmax(out, 1)
        for i in range(indices.shape[0]):
            fake_gt[i, indices[i]] = 1.
        return torch.mean((out-fake_gt)**2)

    def train(self, img, gt=None):
        img = img.to(self.device)
        out = self.net(img)
        if gt is not None:
            gt = gt.to(self.device)
            loss = self.supervised_loss_func(out, gt)
        else:
            loss = self.unsupervised_loss_func(out)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, img):
        img = img.to(self.device)
        self.net.eval()
        with torch.no_grad():
            out = self.net(img)
        self.net.train()
        label = torch.argmax(out, 1).cpu().tolist()
        return label

    def visualize(self, img, save_dir):
        for i in range(10):
            os.makedirs(save_dir+str(i)+'/', exist_ok=True)
        label = self.test(img)
        for i in range(img.shape[0]):
            image = TF.to_pil_image(img[i])
            image.save(save_dir+'%d/%d.jpg'%(label[i], i))

    def test_accuracy(self, img, gt):
        img = img.to(self.device)
        self.net.eval()
        with torch.no_grad():
            out = self.net(img)
        self.net.train()
        out = torch.argmax(out, 1).cpu()
        gt = torch.argmax(gt, 1)
        accuracy = (out == gt).sum().item() / float(out.shape[0])
        return accuracy

    def save(self, save_path):
        save_dir, name = os.path.split(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.net.state_dict(), save_path)

    def load(self, save_path):
        self.net.load_state_dict(torch.load(save_path))

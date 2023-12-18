from networks import pipnet
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torchvision import transforms

def get_meanface(
        meanface_string: str,
        num_nb: int = 10):
    """
    :param meanface_string: a long string contains normalized or un-normalized
     meanface coords, the format is "x0,y0,x1,y1,x2,y2,...,xn-1,yn-1".
    :param num_nb: the number of Nearest-neighbor landmarks for NRM, default 10
    :return: meanface_indices, reverse_index1, reverse_index2, max_len
    """
    meanface = meanface_string.strip("\n").strip(" ").split(" ")
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    meanface_lms = meanface.shape[0]
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i, :]
        dists = np.sum(np.power(pt - meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1 + num_nb])

    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[], []]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            # meanface_indices[i][0,1,2,...,9] -> [[i,i,...,i],[0,1,2,...,9]]
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0] * 10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1] * 10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    # [...,max_len,...,max_len*2,...]
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len, meanface_lms


def normalize_pipnet(img):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = (img - mean[:, None, None]) / std[:, None, None]
    return img

class PIPNet:
    mean_face_string = "0.07960419395480703 0.3921576875344978 0.08315055593117261 0.43509551571809146 0.08675705281580391 0.47810288286566444 0.09141892980469117 0.5210356946467262 0.09839925903528965 0.5637522280060038 0.10871037524559955 0.6060410614977951 0.12314562992759207 0.6475338700558225 0.14242389255404694 0.6877152027028081 0.16706295456951875 0.7259564546408682 0.19693946055282413 0.761730578566735 0.23131827931527224 0.7948205670466106 0.2691730934906831 0.825332081636482 0.3099415030959131 0.853325959406618 0.3535202097901413 0.8782538906229107 0.40089023799272033 0.8984102434399625 0.4529251732310723 0.9112191359814178 0.5078640056794708 0.9146712690731943 0.5616519666079889 0.9094327772020283 0.6119216923689698 0.8950540037623425 0.6574617882337107 0.8738084866764846 0.6994820494908942 0.8482660530943744 0.7388135339780575 0.8198750461527688 0.775158750479601 0.788989141243473 0.8078785221990765 0.7555462713420953 0.8361052138935441 0.7195542055115057 0.8592123871172533 0.6812759034843933 0.8771159986952748 0.6412243940605555 0.8902481006481506 0.5999743595282084 0.8992952868651163 0.5580032282594118 0.9050110573289222 0.5156548913779377 0.908338439928252 0.4731336721500472 0.9104896075281127 0.4305382486815422 0.9124796341441906 0.38798192678294363 0.18465941635742913 0.35063191749632183 0.24110421889338157 0.31190394310826886 0.3003235400132397 0.30828189837331976 0.3603094923651325 0.3135606490643205 0.4171060234289877 0.32433417646045615 0.416842139562573 0.3526729965541497 0.36011177591813404 0.3439660526998693 0.3000863121140166 0.33890077494044946 0.24116055928407834 0.34065620413845005 0.5709736930161899 0.321407825750195 0.6305694459247149 0.30972642336729495 0.6895161625920927 0.3036453838462943 0.7488591859761683 0.3069143844433495 0.8030471337135181 0.3435156012309415 0.7485083446528741 0.3348759588212388 0.6893025057931884 0.33403402013776456 0.6304822892126991 0.34038458762875695 0.5710009285609654 0.34988479902594455 0.4954171902473609 0.40202330022004634 0.49604903449415433 0.4592869389138444 0.49644391662771625 0.5162862508677217 0.4981161256057368 0.5703284628419502 0.40749001573145566 0.5983629921847019 0.4537396729649631 0.6057169923583451 0.5007345777827058 0.6116695615531077 0.5448481727980428 0.6044131443745976 0.5882140504891681 0.5961738788380111 0.24303324896316683 0.40721003719912746 0.27771706732644313 0.3907171413930685 0.31847706697401107 0.38417234007271117 0.3621792860449715 0.3900847721320633 0.3965299162804086 0.41071434661355205 0.3586805562211872 0.4203724421417311 0.31847860588240934 0.4237674602252073 0.2789458001651631 0.41942757306509065 0.5938514626567266 0.4090628827047304 0.6303565516542536 0.3864501652756091 0.6774844732813035 0.3809319896905685 0.7150854850525555 0.3875173254527522 0.747519807465081 0.4025187328459307 0.7155172856447009 0.4145958479293519 0.680051949453018 0.420041513473271 0.6359056750107122 0.41803782782566573 0.33916483987223056 0.6968581311227738 0.40008790639758807 0.6758101185779204 0.47181947887764153 0.6678850445191217 0.5025394453374782 0.6682917934792593 0.5337748367911458 0.6671949030019636 0.6015915330083903 0.6742535357237751 0.6587068892667173 0.6932163943648724 0.6192795131720007 0.7283129162844936 0.5665923267827963 0.7550248076404299 0.5031303335863617 0.7648348885181623 0.4371030429958871 0.7572539606688756 0.3814909500115824 0.7320595346122074 0.35129809553480984 0.6986839074746692 0.4247987356100664 0.69127609583798 0.5027677238758598 0.6911145821740593 0.576997542122097 0.6896269708051024 0.6471352843446794 0.6948977432227927 0.5799932528781817 0.7185288017567538 0.5024914756021335 0.7285408331555782 0.4218115644247556 0.7209126133193829 0.3219750495122499 0.40376441481225156 0.6751136343101699 0.40023415216110797"

    def __init__(self, extend=0.2, num_nb=10, stride=32, num_lms=98, input_size=256, expansion=4, device='cuda:0'):
        self.extend = extend
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.net_stride = stride
        self.input_size = input_size
        self.model = getattr(pipnet, 'PIPNetResNet')(models.resnet101(pretrained=True), expansion, stride, num_lms,
                                                     num_nb)
        self.device = device
        # print('DEVICE',device,torch.cuda.is_available(),torch.cuda.device_count())
        # print('loading : pipnet')
        self.model.load_state_dict(torch.load('./checkpoints/pipnet_resnet101_10x98x32x256_wflw.pth', map_location = self.device))
        
        # print('loaded : pipnet')
        self.model.to(self.device)
        self.max_len = None
        self.reverse_index1 = []
        self.reverse_index2 = []
        self.meanface_indices, self.reverse_index1, \
            self.reverse_index2, self.max_len, self.meanface_lms = get_meanface(meanface_string=self.mean_face_string,
                                                                                num_nb=self.num_nb)
        self.rsz = transforms.Resize((input_size, input_size))

    def __predict(self, image, height, width, start_x, start_y):
        image = image.to(int(self.device.strip(':')[-1]), non_blocking=True)
        device = next(self.model.parameters()).device
        #print("Model device:", device, image.device)
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = self.model(image)
        # (1,68,8,8)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
        assert tmp_batch == 1

        outputs_cls = outputs_cls.view(tmp_batch * tmp_channel, -1)  # (68,64)
        max_ids = torch.argmax(outputs_cls, 1)  # (68,)
        max_ids = max_ids.view(-1, 1)  # (68,1)
        max_ids_nb = max_ids.repeat(1, self.num_nb).view(-1, 1)  # (68,10) -> (68*10,1)

        outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)  # (68,64)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)  # (68,1)
        outputs_x_select = outputs_x_select.squeeze(1)  # (68,)
        outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)
        outputs_y_select = outputs_y_select.squeeze(1)  # (68,)

        outputs_nb_x = outputs_nb_x.view(tmp_batch * self.num_nb * tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)  # (68*10,1)
        outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, self.num_nb)  # (68,10)
        outputs_nb_y = outputs_nb_y.view(tmp_batch * self.num_nb * tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, self.num_nb)  # (68,10)

        lms_pred_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_x_select.view(-1, 1)  # x=cx+offset_x
        lms_pred_y = torch.floor(max_ids / tmp_width).view(-1, 1).float() + outputs_y_select.view(-1,
                                                                                                  1)  # y=cy+offset_y
        lms_pred_x /= 1.0 * self.input_size / self.net_stride  # normalize coord (x*32)/256
        lms_pred_y /= 1.0 * self.input_size / self.net_stride  # normalize coord (y*32)/256

        lms_pred_nb_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_nb_x_select  # (68,10)
        lms_pred_nb_y = torch.floor(max_ids / tmp_width).view(-1, 1).float() + outputs_nb_y_select  # (68,10)
        lms_pred_nb_x = lms_pred_nb_x.view(-1, self.num_nb)  # (68,10)
        lms_pred_nb_y = lms_pred_nb_y.view(-1, self.num_nb)  # (68,10)
        lms_pred_nb_x /= 1.0 * self.input_size / self.net_stride  # normalize coord (nx*32)/256
        lms_pred_nb_y /= 1.0 * self.input_size / self.net_stride  # normalize coord (ny*32)/256

        # merge predictions
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(self.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(self.num_lms, self.max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)

        # de-normalize
        lms_pred_merge[:, 0] *= width  # e.g 256
        lms_pred_merge[:, 1] *= height  # e.g 256
        lms_pred_merge[:, 0] += start_x  # e.g 256
        lms_pred_merge[:, 1] += start_y
        image.detach().cpu()
        return lms_pred_merge.detach().cpu().numpy().tolist()

    def __crop_faces(self, frames, faces, landmarks):
        for j, (image, bboxes) in enumerate(zip(frames, faces)):
            _, height, width = image.shape
            for i, d in enumerate(bboxes):
                # print(dprint
                if d == [0, 0, 0, 0]:
                    landmarks[j].append([[0, 0]])
                    continue
                x1, y1, x2, y2 = d
                # print(x1, y1, x2, y2)
                w = x2 - x1 + 1
                h = y2 - y1 + 1

                x1 -= int(w * (1. + self.extend - 1) / 2)
                y1 += int(h * (1. + self.extend - 1) / 2)
                x2 += int(w * (1. + self.extend - 1) / 2)
                y2 += int(h * (1. + self.extend - 1) / 2)
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, width - 1)
                y2 = min(y2, height - 1)


                # print(y1, y2, x1, x2, image.shape)
                # Crop the faces using PyTorch operations
                cropped = image[:, y1:y2, x1:x2]
                _, crop_h, crop_w = cropped.shape

                # Resize the cropped faces to the desired input size
                cropped = self.rsz(cropped)

                # Perform any additional processing on the cropped faces, if needed

                # Normalize the cropped face using the 'normalize_pipnet' function
                cropped = normalize_pipnet(img=cropped).contiguous().unsqueeze(0)
                lms = self.__predict(cropped, crop_h, crop_w, x1, y1)
                landmarks[j] = lms
        return landmarks

    def __call__(self, frames, faces):
        # print([[]]*len(frames))
        return self.__crop_faces(frames, faces, [[[]]]*len(frames))

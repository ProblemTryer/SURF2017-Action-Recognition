from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_data


def main():
    # load train dataset
    print '@@@@@@@@@@@@@'
    data = load_data(data_path='./data', split='train')
    label_to_idx = data['label_to_idx']
    print '@@@@@'
    # load val dataset to print out bleu scores every epoch
    #val_data = load_data(data_path='./data', split='val')
    val_data = []

    print label_to_idx
    print data['labels']

    model = CaptionGenerator(label_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=30, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=1, batch_size=1, update_rule='adam',
                                          learning_rate=0.001, print_every=3, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-10',
                                     print_bleu=True, log_path='log/')
    print 'model + solver'

    solver.train()

if __name__ == "__main__":
    main()
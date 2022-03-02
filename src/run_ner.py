# -*- coding: utf-8 -*-

import gc
import time
import warnings
from argparse import ArgumentParser

from tqdm import tqdm

from src.utils.bert_utils import *
from src.utils.ner_utils import *
from src.utils.utils import save_pickle, load_pkl, load_file, save_pkl


def read_data(args, tokenizer, label_list):

    train_inputs, dev_inputs = defaultdict(list), defaultdict(list)

    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            text = words.split('\002')
            label = labels.split('\002')
            build_bert_inputs(train_inputs, label, text, tokenizer, label_list)

    with open(args.dev_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            text = words.split('\002')
            label = labels.split('\002')
            build_bert_inputs(dev_inputs, label, text, tokenizer, label_list)

    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    save_pickle(train_inputs, train_cache_pkl_path)
    save_pickle(dev_inputs, dev_cache_pkl_path)


def train(args):

    label_pkl_path = os.path.join(args.output_path, "label_list.pkl")
    if os.path.exists(label_pkl_path):
        label_list = load_pkl(label_pkl_path)
    else:
        tokens_list = load_file(args.origin_train_path, sep=args.sep)
        label_list = set([tokens[1] for tokens in tokens_list if len(tokens) == 2])

        label_list = set(sorted(list(label_list)))

    if len(label_list) == 0:
        ValueError("loading labels error, labels type not found in data file: {}".format(args.output_path))
    else:
        save_pkl(label_list, label_pkl_path)

    label2id_path = os.path.join(args.output_path, "label2id.pkl")
    if os.path.exists(label2id_path):
        label2id = load_pkl(label2id_path)
    else:
        label2id = {l: i for i, l in enumerate(label_list)}
        save_pkl(label2id, label2id_path)

    id2label = {value: key for key, value in label2id.items()}

    num_labels = len(label_list)

    tokenizer, model = build_model_and_tokenizer(args, num_labels)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer, label2id)

    train_dataloader, dev_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)
    optimizer, scheduler = build_optimizer(args, model, total_steps)

    global_steps, total_loss, cur_avg_loss, best_f1 = 0, 0., 0., 0.

    print("\n >> Start training ... ... ")
    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Epoch : {epoch}', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:

            model.zero_grad()

            batch_cuda = batch2cuda(args, batch)
            input_ids, _, token_type_ids, attention_mask, label_ids = batch_cuda
            loss = model(input_ids, token_type_ids, attention_mask, label_ids)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if args.use_fgm:
                fgm = FGM(args, model)
                fgm.attack()
                adv_loss = model(input_ids, token_type_ids, attention_mask, label_ids)
                adv_loss.backward()
                fgm.restore()

            if args.use_pgd:
                pgd = PGD(args, model)
                pgd.backup_grad()
                for t in range(args.adv_k):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_loss = model(input_ids, token_type_ids, attention_mask, label_ids)
                    adv_loss.backward()
                pgd.restore()

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if args.use_ema:
                if args.ema_start:
                    ema.update()

            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_steps == 0:

                epoch_avg_loss = cur_avg_loss / args.logging_steps
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                if args.use_ema:
                    if global_steps >= args.ema_start_step and not args.ema_start:
                        print('\n>>> EMA starting ...')
                        args.ema_start = True
                        ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.95)

                if args.do_eval:

                    if args.use_ema:
                        if args.ema_start:
                            ema.apply_shadow()

                    print("\n >> Start evaluating ... ... ")

                    overall, by_type = evaluate(args, dev_dataloader, model, id2label)

                    f1_score = overall.fscore
                    f1_score = round(f1_score, 4)

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        save_model(args, model, tokenizer)
                        print(f"\n >> Best saved, f1 is {f1_score} !")

                    if args.use_ema:
                        if args.ema_start:
                            ema.restore()

                    model.train()
                    cur_avg_loss = 0.

            global_steps += 1
            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

    if args.use_ema:
        ema.apply_shadow()

    if not args.do_eval:
        save_model(args, model, tokenizer)

    data = time.asctime(time.localtime(time.time())).split(' ')
    now_time = data[-1] + '-' + data[-5] + '-' + data[-3] + '-' + \
    data[-2].split(':')[0] + '-' + data[-2].split(':')[1] + '-' + data[-2].split(':')[2]
    os.makedirs(os.path.join(args.output_path, f'f1-{best_f1}-{now_time}'), exist_ok=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    print('\n >> Finish training .')


def evaluate(args, dev_dataloader, model, id2label):

    all_origin_tokens, all_origin_labels, all_predict_labels = [], [], []
    dev_iterator = tqdm(dev_dataloader, desc='Evaluating', total=len(dev_dataloader))

    with torch.no_grad():
        for batch in dev_iterator:
            batch_cuda = batch2cuda(args, batch)
            input_ids, origin_tokens, token_type_ids, attention_mask, label_ids = batch_cuda
            crf_predict = model.predict(input_ids, token_type_ids, attention_mask)

            origin_tokens = batch['origin_tokens']
            for i in origin_tokens:
                all_origin_tokens.append([j for j in i])

            for l in crf_predict:
                all_predict_labels.append([id2label[idx] for idx in l])

            for l in label_ids:
                all_origin_labels.append([id2label[idx.item()] for idx in l])

    eval_list = []
    for origin_tokens, origin_labels, predict_labels in \
            zip(all_origin_tokens, all_origin_labels, all_predict_labels):
        for ot, ol, pl in \
                zip(origin_tokens, origin_labels, predict_labels):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")

    counts = evaluation(eval_list)
    report(counts)

    overall, by_type = metrics(counts)

    return overall, by_type


def main():
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--output_path', type=str,
                        default=f'../user_data/output_model')
    parser.add_argument('--submit_path', type=str,
                        default='../submita.csv')
    parser.add_argument('--origin_train_path', type=str,
                        default=f'../raw_data/train_500.txt')
    parser.add_argument('--train_path', type=str,
                        default=f'../raw_data/train1.json')
    parser.add_argument('--dev_path', type=str,
                        default=f'../raw_data/dev1.json')
    parser.add_argument('--data_cache_path', type=str,
                        default=f'../user_data/process_data/pkl')

    parser.add_argument('--model_name_or_path', type=str,
                        default=f'../user_data/pretrain_model/bert-base-chinese')

    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--sep', type=str, default=' ')

    parser.add_argument('--do_eval', type=bool, default=True)

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=91)

    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)

    parser.add_argument('--dropout_rate', type=float, default=0.1)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--crf_learning_rate', type=float, default=0.1)
    parser.add_argument('--other_learning_rate', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--use_fgm', type=bool, default=False)
    parser.add_argument('--use_pgd', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=False)
    parser.add_argument('--use_lookahead', type=bool, default=False)

    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')

    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--ema_start_step', type=int, default=0)

    parser.add_argument('--logging_steps', type=int, default=100)

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    path_list = [args.output_path, args.data_cache_path]
    for i in path_list:
        os.makedirs(i, exist_ok=True)

    seed_everything(args.seed)

    train(args)


if __name__ == '__main__':
    main()

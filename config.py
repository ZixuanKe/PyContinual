import copy
import shutil
import argparse
import logging

from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=16,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )


    parser.add_argument("--idrandom", type=int, help="which sequence to use")
    parser.add_argument("--pt_task", type=int, help="task id")
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument("--baseline", type=str,default='', help="one,ncl")
    parser.add_argument('--task_name', type=str)
    parser.add_argument("--saved_output_dir", type=str, default='./ckpt', help="Where to store the final model.")
    parser.add_argument("--sequence_file",type=str, help="sequence file")
    parser.add_argument('--thres_cosh',default=50,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--thres_emb',default=6,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--lamb',type=float,required=False,help='(default=%(default)d)')
    parser.add_argument("--base_dir", default='/hdd_3/zke4',type=str, help="task id")
    parser.add_argument("--eval_checkpoint",action="store_true")
    parser.add_argument("--no_repeat_ngram_size", type=int, help="task id")
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_min_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument("--num_train_epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )

    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument('--pool_size', type=int)
    parser.add_argument('--prompt_lr', type=float)
    parser.add_argument('--adapter_lr', type=float)
    parser.add_argument('--classifier_lr', type=float)
    parser.add_argument('--share_lr', type=float)
    parser.add_argument('--patient', default=5, type=int)
    parser.add_argument('--warmup_ratio',  type=float)
    parser.add_argument('--dataset_type', default='full', type=str)
    parser.add_argument('--prompt_position', default='front', type=str)
    parser.add_argument('--mixed_precision',type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--lamb_m', default=1, type=float)
    parser.add_argument('--supsup_sparsity', default=0.9, type=float)
    parser.add_argument('--reduction_factor', default=64, type=float)
    parser.add_argument('--teacher_decoder_hidden_states', type=list)
    parser.add_argument('--teacher_encoder_hidden_states', type=list)
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--sample_cap', type=float)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--sample_num_per_class',  type=int)
    parser.add_argument(
        "--per_device_train_pool_batch_size",
        default=8,
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--only_eval_current_task",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument('--eps', default=0, type=float) #-1e-4, when it is the same, we cannot judage, choose not to open
    parser.add_argument('--semantic_cap_size',default=3,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--num_semantic_cap',default=3,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument("--smax", default=400, type=int, help="smax")
    parser.add_argument("--n_tokens", default=20, type=int, help="prompt token length")
    parser.add_argument('--M', default=10, type=int)
    parser.add_argument('--N', default=5, type=int)
    parser.add_argument('--Lp', default=12, type=int)
    parser.add_argument(
        "--disable_impt_comparison",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--known_similarity",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        '--buffer_size_per_dataset',
        type=int,
        default=50,
        help="buffer size for each dataset",
    )
    parser.add_argument(
        '--replay_freq',
        type=int,
        default=1,
        help='replay frequency.'
    )
    parser.add_argument('--replay_beta', default=0.03, type=float, help='(default=%(default)f)')
    parser.add_argument('--replay_alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--mer_beta', default=0.03, type=float, help='(default=%(default)f)')
    parser.add_argument('--mer_gamma', default=1.0, type=float, help='(default=%(default)f)')
    parser.add_argument('--grad_clip_norm', default=1.0, type=float, help='(default=%(default)f)')
    args = parser.parse_args()

    return args



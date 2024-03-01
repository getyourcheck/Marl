tokenize_policy = 'chatml'

token_cfg = dict(
    reward_score_id=99432,
    role_cfg=dict(
        system=dict(
            begin=dict(
                with_name='<TOKENS_UNUSED_140>system name={name}\n',
                without_name='<TOKENS_UNUSED_140>system\n',
                name={
                    'interpreter': '<TOKENS_UNUSED_136>',
                    'plugin': '<TOKENS_UNUSED_135>',
                }),
            end='<TOKENS_UNUSED_139>\n',
            loss=dict(
                meta=False,
                icl=False,
                current=False,
                prefix=False,
            )),
        user=dict(
            begin=dict(
                with_name='<TOKENS_UNUSED_140>user name={name}\n',
                without_name='<TOKENS_UNUSED_140>user\n',
            ),
            end='<TOKENS_UNUSED_139>\n',
            loss=dict(
                icl=False,
                current=False,
                prefix=False,
            )),
        assistant=dict(
            begin=dict(
                with_name='<TOKENS_UNUSED_140>assistant name={name}\n',
                without_name='<TOKENS_UNUSED_140>assistant\n',
                name={
                    'interpreter': '<TOKENS_UNUSED_136>',
                    'plugin': '<TOKENS_UNUSED_135>',
                }),
            end='<TOKENS_UNUSED_139>\n',
            loss=dict(
                icl=True,
                current=True,
                prefix=False,
                end=True,
            )),
        environment=dict(
            begin=dict(
                with_name='<TOKENS_UNUSED_140>environment name={name}\n',
                without_name='<TOKENS_UNUSED_140>environment\n',
                name={
                    'interpreter': '<TOKENS_UNUSED_136>',
                    'plugin': '<TOKENS_UNUSED_135>',
                }),
            end='<TOKENS_UNUSED_139>\n',
            loss=dict(
                icl=False,
                current=False,
                prefix=False,
            )),
        tool=dict(
            begin=dict(
                with_name='<TOKENS_UNUSED_138>{name}\n',
                name={
                    'interpreter': '<TOKENS_UNUSED_136>',
                    'plugin': '<TOKENS_UNUSED_135>',
                }),
            end='<TOKENS_UNUSED_137>\n',
            belong='assistant',
        ),
        thought=dict(
            begin=dict(without_name=''),
            end='',
            belong='assistant',
        ),
    ),
)

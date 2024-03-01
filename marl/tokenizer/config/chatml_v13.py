tokenize_policy = 'chatml'

token_cfg = dict(
    reward_score_id=92527,
    role_cfg=dict(
        system=dict(
            begin=dict(
                with_name='[UNUSED_TOKEN_146]system name={name}\n',
                without_name='[UNUSED_TOKEN_146]system\n',
                name={
                    'interpreter': '[UNUSED_TOKEN_142]',
                    'plugin': '[UNUSED_TOKEN_141]',
                }),
            end='[UNUSED_TOKEN_145]\n',
            loss=dict(
                meta=False,
                icl=False,
                current=False,
                prefix=False,
            )),
        user=dict(
            begin=dict(
                with_name='[UNUSED_TOKEN_146]user name={name}\n',
                without_name='[UNUSED_TOKEN_146]user\n',
            ),
            end='[UNUSED_TOKEN_145]\n',
            loss=dict(
                icl=False,
                current=False,
                prefix=False,
            )),
        assistant=dict(
            begin=dict(
                with_name='[UNUSED_TOKEN_146]assistant name={name}\n',
                without_name='[UNUSED_TOKEN_146]assistant\n',
                name={
                    'interpreter': '[UNUSED_TOKEN_142]',
                    'plugin': '[UNUSED_TOKEN_141]',
                }),
            end='[UNUSED_TOKEN_145]\n',
            loss=dict(
                icl=True,
                current=True,
                prefix=False,
                end=True,
            )),
        environment=dict(
            begin=dict(
                with_name='[UNUSED_TOKEN_146]environment name={name}\n',
                without_name='[UNUSED_TOKEN_146]environment\n',
                name={
                    'interpreter': '[UNUSED_TOKEN_142]',
                    'plugin': '[UNUSED_TOKEN_141]',
                }),
            end='[UNUSED_TOKEN_145]\n',
            loss=dict(
                icl=False,
                current=False,
                prefix=False,
            )),
        tool=dict(
            begin=dict(
                with_name='[UNUSED_TOKEN_144]{name}\n',
                name={
                    'interpreter': '[UNUSED_TOKEN_142]',
                    'plugin': '[UNUSED_TOKEN_141]',
                }),
            end='[UNUSED_TOKEN_143]\n',
            belong='assistant',
        ),
        thought=dict(
            begin=dict(without_name=''),
            end='',
            belong='assistant',
        ),
    ),
)

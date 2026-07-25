"""Add user table and user_id to conversations

Revision ID: 002
Revises: 001
Create Date: 2026-07-23 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_admin', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)

    # Add user_id to conversations
    op.add_column('conversations', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_conversations_user_id'), 'conversations', ['user_id'], unique=False)
    op.create_index('idx_conversation_user_updated', 'conversations', ['user_id', 'updated_at'], unique=False)

    # Add user_id to long_term_memory
    op.add_column('long_term_memory', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_long_term_memory_user_id'), 'long_term_memory', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_long_term_memory_user_id'), table_name='long_term_memory')
    op.drop_column('long_term_memory', 'user_id')

    op.drop_index('idx_conversation_user_updated', table_name='conversations')
    op.drop_index(op.f('ix_conversations_user_id'), table_name='conversations')
    op.drop_column('conversations', 'user_id')

    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_table('users')

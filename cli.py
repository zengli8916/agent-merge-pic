import logging
import sys
from pathlib import Path

import click

from agent import LivestreamAgent
from models import Requirements


@click.command()
@click.option("--reference", "-r", required=True, type=click.Path(exists=True), help="参考直播间图片路径")
@click.option("--product", "-p", multiple=True, required=True, type=click.Path(exists=True), help="产品图片路径，可多次使用")
@click.option("--person", required=True, help="人物描述，例如：马来西亚年轻女性")
@click.option("--background", required=True, help="背景描述，例如：保健品货柜")
@click.option("--note", "-n", multiple=True, help="补充要求，可多次使用。例如：-n '瓶子大小一样'")
@click.option("--max-rounds", default=5, type=int, help="最大迭代总轮次（默认5）")
@click.option("--restart-after", default=5, type=int, help="连续失败多少轮后重新生成（默认5，即不重置）")
@click.option("--output", "-o", default="output", type=click.Path(), help="输出目录")
@click.option("--verbose", "-v", is_flag=True, help="详细日志输出")
def generate(reference: str, product: tuple, person: str, background: str, note: tuple, max_rounds: int, restart_after: int, output: str, verbose: bool):
    """生成电商直播间卖货图片"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)

    from config import API_KEY
    if not API_KEY:
        click.echo("Error: GEMINI_API_KEY environment variable not set", err=True)
        sys.exit(1)

    notes = "\n".join(f"- {n}" for n in note) if note else ""
    description = f"- 人物：{person}\n- 背景：{background}"
    if notes:
        description += f"\n- 补充要求：\n{notes}"

    requirements = Requirements(description=description)
    product_list = list(product)
    agent = LivestreamAgent(output_dir=output, max_rounds=max_rounds, max_retries_before_restart=restart_after)

    click.echo(f"Starting generation...")
    click.echo(f"  Reference: {reference}")
    click.echo(f"  Products: {', '.join(product_list)}")
    click.echo(f"  Person: {person}")
    click.echo(f"  Background: {background}")
    if note:
        click.echo(f"  Notes: {'; '.join(note)}")
    click.echo(f"  Output: {output}")
    click.echo()

    result = agent.run(reference, product_list, requirements)

    click.echo()
    if result.partial:
        click.echo(f"Completed (partial): best result after {result.rounds} rounds")
    else:
        click.echo(f"Completed: passed at round {result.rounds}")

    click.echo(f"Output: {result.image_path}")

    if result.evaluation:
        scores = result.evaluation.scores
        click.echo(f"Scores: pose={scores.pose_realism} product={scores.product_accuracy} "
                   f"size={scores.product_size} match={scores.requirements_match} "
                   f"physics={scores.physics}")


if __name__ == "__main__":
    generate()

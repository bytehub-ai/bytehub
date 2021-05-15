import click
import bytehub as bh


@click.group()
def cli():
    pass


@cli.command()
@click.argument("task_name")
def run(task_name):
    """Run a ByteHub task."""
    click.echo("Connecting to Feature Store...")
    fs = bh.FeatureStore()
    if "/" not in task_name:
        raise click.UsageError("Task name must be specified as namespace/name")
    try:
        fs.run_task(task_name)
    except Exception as e:
        raise click.ClickException(f"Failed to run {task_name}: {e}")

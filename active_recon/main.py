from utils.config import load_parser
from task.task import Task

if __name__ == "__main__":
    args = load_parser()
    active_recon_task = Task(args)

    active_recon_task.run(args)

    active_recon_task.save()

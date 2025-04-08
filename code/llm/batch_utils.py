# Utility functions to handle batch jobs
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
client = OpenAI()

batch_ids = [
    'batch_67f503e24c808190ae40cf1c29982b68',
    'batch_67f503e79424819087db3e0715ed0206',
    'batch_67f503eb04908190bfbeda23b1c167d3',
    'batch_67f503efa914819094f769a0cad012d3',
    'batch_67f503f4d86081909fb1d5f824fcd9b6',
    'batch_67f503f872d0819082d2942248223618',
    'batch_67f503fd43688190b102eb74769e354d',
    'batch_67f50401a9bc8190bba9498db83af4b2',
    'batch_67f504065f188190b6b34168728803d9',
    'batch_67f5040bf1708190b9da5ef7a7541236',
]

def query_batch_status():
    # query status of all batch jobs in {batch_ids}
    for id in batch_ids:
        status = client.batches.retrieve(id)
        print(f"\nBatch job {id} is {status.status}:\n{status}\n")
        print('-' * 50)

def cancel_batch_job(id):
    # cancel batch job with id
    client.batches.cancel(id)
    print(f"Batch job {id} cancelled.")

def download_batch_output(id, out_dir):
    # download batch job output with id
    status = client.batches.retrieve(id)
    if status.status != 'completed':
        print(f"Batch job {id} not completed yet.")
        return
    file_id = status.output_file_id
    file_response = client.files.content(file_id)
    
    with open(f"{out_dir.strip('/') if out_dir else '.'}/batch_output_{id.strip('batch_')}.jsonl", 'w') as f:
        f.write(file_response.text)
    print(f"Batch job {id} output downloaded.")


if __name__ == '__main__':
    query_batch_status()
    # cancel_batch_job('batch_123456')
    # download_batch_output('batch_67f4d404858881908824a158bafc613e', 'data/science')
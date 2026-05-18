import os
import sys
try:
    from MessageClient.ProgressMessageSender import ProgressMessageSender
except Exception:
    ProgressMessageSender = None


def send(progress, status, info):
    bootstrap_servers = os.getenv('KAFKA_SERVER_IP_PORT')
    topic = os.getenv('KAFKA_TOPIC')
    task_id = os.getenv('KAFKA_TASK_ID')

    if not ProgressMessageSender or not bootstrap_servers or not topic:
        print(f"[Log Only] {progress}% - {status} - {info}", flush=True)
        return

    try:
        sender = ProgressMessageSender(bootstrap_servers, topic, task_id)
        if hasattr(sender, 'is_none') and sender.is_none():
            print(f"[Log Only] {progress}% - {status} - {info}", flush=True)
            return
        message_dict = {
            'progress': int(progress),
            'runningStatus': status,
            'runningInfo': info,
            'success': 1 if status == 'completed' else 0
        }
        sender.send(message_dict)
        print(f"[Kafka] {progress}% - {status} - {info}", flush=True)
    except Exception as e:
        print(f"[Kafka failed] {e}  msg: {progress}% - {status} - {info}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        send(sys.argv[1], sys.argv[2], sys.argv[3])

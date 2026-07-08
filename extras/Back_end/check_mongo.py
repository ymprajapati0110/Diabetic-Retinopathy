import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def ping():
    client = AsyncIOMotorClient('mongodb://localhost:27017', serverSelectionTimeoutMS=2000)
    try:
        info = await client.server_info()
        print("Connected successfully to MongoDB!")
        print(info)
    except Exception as e:
        print("Failed to connect to MongoDB:")
        print(e)
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(ping())

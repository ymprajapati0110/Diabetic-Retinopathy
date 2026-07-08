import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def main():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['dr_medical_ai']
    
    # List all users
    users = await db.users.find({}).to_list(length=100)
    print("=== ALL USERS ===")
    for u in users:
        print(f"  Email: {u['email']} | Role: {u['role']}")
    
    # Verify ALL pending users
    result = await db.users.update_many(
        {"role": "pending"},
        {"$set": {"role": "verified"}}
    )
    print(f"\n=== VERIFIED {result.modified_count} PENDING ACCOUNT(S) ===")
    
    # Confirm
    users = await db.users.find({}).to_list(length=100)
    print("\n=== UPDATED USERS ===")
    for u in users:
        print(f"  Email: {u['email']} | Role: {u['role']}")
    
    client.close()

asyncio.run(main())

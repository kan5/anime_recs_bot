from my_utils import client
import asyncio

db = client['anime_recs_bot']
history = db['messages']
# print(await db.test_collection.find_one({'i': {'$lt': 1}}))

async def main():
    print(await db.test_collection.delete_one({'from': {'ts': 1, 'username': 'boba'}}))
    # await asyncio.sleep(5)
    # print(await db.test_collection.find_one({}))
    async def do_find():
        cursor = db.test_collection.find().sort('i')
        for document in await cursor.to_list(length=100):
            print(document)
    await do_find()
    print(db.test_collection)

if __name__ == '__main__':
    asyncio.run(main())
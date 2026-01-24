pragma solidity ^0.8.0;

contract UpdateLog {

    struct Update {
        string clientId;
        string modelHash;
        uint timestamp;
    }

    Update[] public updates;

    function logUpdate(string memory cid, string memory h) public {
        updates.push(Update(cid, h, block.timestamp));
    }

    function totalUpdates() public view returns(uint) {
        return updates.length;
    }
}

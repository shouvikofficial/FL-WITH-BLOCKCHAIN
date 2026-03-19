// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UpdateLog {

    // ======================================================
    // DATA STRUCTURE
    // ======================================================
    struct Update {
        string  clientId;
        string  modelHash;
        uint256 timestamp;
        uint256 roundId;
    }

    Update[] public updates;

    // ======================================================
    // EVENTS (viewable in transaction logs)
    // ======================================================
    event UpdateLogged(
        string  indexed clientId,
        string  modelHash,
        uint256 roundId,
        uint256 timestamp
    );

    // ======================================================
    // WRITE — log a federated learning update
    // ======================================================
    function logUpdate(string memory cid, string memory h) public {
        updates.push(Update({
            clientId:   cid,
            modelHash:  h,
            timestamp:  block.timestamp,
            roundId:    updates.length + 1
        }));

        emit UpdateLogged(cid, h, updates.length, block.timestamp);
    }

    // ======================================================
    // READ — total number of logged updates
    // ======================================================
    function totalUpdates() public view returns (uint256) {
        return updates.length;
    }

    // ======================================================
    // READ — get a specific update by index
    // ======================================================
    function getUpdate(uint256 index) public view returns (
        string  memory clientId,
        string  memory modelHash,
        uint256        timestamp,
        uint256        roundId
    ) {
        require(index < updates.length, "Index out of bounds");
        Update memory u = updates[index];
        return (u.clientId, u.modelHash, u.timestamp, u.roundId);
    }
}
